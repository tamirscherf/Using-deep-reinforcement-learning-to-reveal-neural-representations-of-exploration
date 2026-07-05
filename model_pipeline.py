"""Deep reinforcement learning model pipeline.

Restless bandit environment, LSTM actor-critic (A2C) training, sine value functions,
and inference on participant trajectories (activations, probabilities, likelihoods).
"""

from __future__ import annotations

import copy
import itertools
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Model / task / training configuration.
# --------------------------------------------------------------------------- #
CFG = {
    "task": {"num_actions": 3, "num_trials": 200},
    "net": {"hidden_size": 48},
    "train": {
        "gamma": 0.75,                    # discount factor
        "beta_critic": [0.05, 0.05],      # critic loss weight (per distribution)
        "beta_entropy": [1.0, 1.0],       # entropy regularization weight
        "optimizer_params": {"lr": 0.001},
        "scheduler_params": {"step_size": 100_000, "gamma": 0.1},
        "use_scheduler": 1,
    },
    "force_cpu": True,
}

import paths

device = "cuda" if torch.cuda.is_available() and not CFG["force_cpu"] else "cpu"

LOGS_DIR = paths.LOGS_DIR
SIN_FUNC_DIR = paths.SIN_FUNC_DIR
TRAINED_MODEL_NAME = paths.TRAINED_MODEL_NAME


# ========================================================================== #
# Weight initialization helpers
# ========================================================================== #
def init_weights_(module: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal initialization for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def double_unsqz(x: torch.Tensor) -> torch.Tensor:
    """Add a leading (sequence, batch) pair of singleton dims for the LSTM."""
    x.unsqueeze_(0)
    x.unsqueeze_(0)
    return x


# ========================================================================== #
# Environment: restless three-armed bandit
# ========================================================================== #
class RestlessBandit:
    """Simulates one episode of the restless multi-armed bandit task.

    ``sin_data`` has shape (n_combinations, n_arms, n_trials); each trial the
    state is [trial_index, previous_reward, one-hot previous action].
    """

    def __init__(self, sin_data: np.ndarray, max_rew: np.ndarray):
        self.num_cmb, self.num_arms, self.num_trials = sin_data.shape
        self.sin_data = sin_data
        self.max_rew = max_rew
        self.rng = np.random.default_rng()
        self.current_dist_idx = 0  # single reward distribution
        self.reset()

    def _initial_state(self) -> torch.Tensor:
        initial_reward = torch.zeros(1, dtype=torch.float32, device=device)
        initial_action = torch.zeros(self.num_arms, dtype=torch.float32, device=device)
        initial_trial = torch.zeros(1, dtype=torch.float32, device=device)
        state = torch.cat([initial_trial, initial_reward, initial_action], 0)
        return double_unsqz(state)

    def reset(self):
        self.trial = 0
        self.current_epsd_idx = np.random.choice(self.num_cmb)
        self.epsd_sins = copy.deepcopy(self.sin_data[self.current_epsd_idx])
        epsd_max_rew = self.max_rew[self.current_epsd_idx]
        self.rng.shuffle(self.epsd_sins, axis=0)  # shuffle arm assignment
        return self._initial_state(), epsd_max_rew

    def step(self, action: torch.Tensor):
        reward = self.epsd_sins[action, self.trial]
        self.trial += 1
        t_reward = torch.tensor([reward], dtype=torch.float32, device=device)
        one_hot_action = F.one_hot(action, num_classes=self.num_arms).squeeze()
        t_trial = torch.tensor([self.trial], dtype=torch.float32, device=device)
        state = torch.cat([t_trial, t_reward, one_hot_action], 0)
        return double_unsqz(state), reward

    def get_dist_idx(self) -> int:
        return self.current_dist_idx


class RestlessBanditRewPun(RestlessBandit):
    """Bandit that randomly draws Gain (reward) or Loss (punishment) episodes."""

    def __init__(self, sin_data, max_rew, min_pun):
        self.min_pun = min_pun
        super().__init__(sin_data, max_rew)

    def reset(self):
        self.trial = 0
        epsd_idx = np.random.choice(self.num_cmb)
        self.current_dist_idx = np.random.randint(0, 2)  # 0 = reward, 1 = punishment
        if self.current_dist_idx == 0:
            self.epsd_sins = copy.deepcopy(self.sin_data[epsd_idx])
            epsd_max = self.max_rew[epsd_idx]
        else:
            self.epsd_sins = -copy.deepcopy(self.sin_data[epsd_idx])
            epsd_max = self.min_pun[epsd_idx]
        self.rng.shuffle(self.epsd_sins, axis=0)
        return self._initial_state(), epsd_max


# ========================================================================== #
# Model: LSTM + actor + critic (A2C)
# ========================================================================== #
class LstmModule(nn.Module):
    def __init__(self):
        super().__init__()
        # input: trial(1) + previous reward(1) + one-hot previous action(n_actions)
        self.input_size = 2 + CFG["task"]["num_actions"]
        self.num_units = CFG["net"]["hidden_size"]
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.num_units)

    def forward(self, state, hidden_states):
        return self.lstm(state, hidden_states)


class ActorModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_units = CFG["net"]["hidden_size"]
        self.num_actions = CFG["task"]["num_actions"]
        self.actor = nn.Linear(self.num_units, self.num_actions)
        init_weights_(self.actor)

    def forward(self, x):
        return nn.functional.log_softmax(self.actor(x), dim=2)


class CriticModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_units = CFG["net"]["hidden_size"]
        self.critic = nn.Linear(self.num_units, 1)
        init_weights_(self.critic, gain=0.1)

    def forward(self, x):
        return self.critic(x)


class A2C(nn.Module):
    """Actor-critic model sharing a common LSTM backbone."""

    def __init__(self, lstm: LstmModule, actor: ActorModule, critic: CriticModule):
        super().__init__()
        self.lstm = lstm
        self.actor = actor
        self.critic = critic
        self.input_size = lstm.input_size
        self.num_units = lstm.num_units
        self.num_actions = actor.num_actions

    def forward(self, state, hidden_states):
        lstm_output, hidden_states = self.lstm(state, hidden_states)
        actions_log_prob = self.actor(lstm_output)
        value = self.critic(lstm_output)
        return actions_log_prob, value, hidden_states


def create_A2C() -> A2C:
    return A2C(LstmModule(), ActorModule(), CriticModule())


def _zero_hidden():
    h = torch.zeros(1, 1, CFG["net"]["hidden_size"], dtype=torch.float32, device=device)
    c = torch.zeros(1, 1, CFG["net"]["hidden_size"], dtype=torch.float32, device=device)
    return h, c


# ========================================================================== #
# Checkpoint I/O
# ========================================================================== #
def save_checkpoint(fname, model, hidden_state, optimizer=None, verbose=False):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    torch.save(
        {
            "model": model,
            "hidden_state": hidden_state,
            "state_dict": model.state_dict(),
            "optimizer": optimizer,
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "cfg": CFG,
        },
        fname,
    )
    if verbose:
        print("Saved model to:", fname)


def load_checkpoint(fname):
    """Load a saved A2C model checkpoint."""
    assert os.path.isfile(fname), f"Missing checkpoint: {fname}"
    items = torch.load(fname, map_location=device)
    model = items["model"]
    model.load_state_dict(items["state_dict"])
    return model, items.get("hidden_state"), items.get("optimizer")


def load_trained_model(model_name: str = TRAINED_MODEL_NAME):
    """Load the trained model checkpoint."""
    model, hidden_states, _ = load_checkpoint(paths.TRAINED_MODEL_PATH)
    model.eval()
    return model, hidden_states


# ========================================================================== #
# Sine value-function generation (training/test stimuli)
# ========================================================================== #
NUM_TRIALS = 200
NUM_ARMS = 3
AMPLITUDE = np.array([20, 20, 20])
NOISE_SD = 5


def create_sin_value_functions(periods_per_session=(3, 2, 6),
                               period_shift=(0.1, 0.45, 0.9),
                               offset=50) -> np.ndarray:
    """Generate one set of per-arm sinusoidal value functions (arms x trials)."""
    periods = np.array(periods_per_session)
    shifts = np.array(period_shift)
    trials_per_period = NUM_TRIALS / periods
    shift_value = (shifts * trials_per_period) * 2 * np.pi * periods / NUM_TRIALS

    y = np.zeros((NUM_ARMS, NUM_TRIALS))
    for t in range(NUM_TRIALS):
        for c in range(NUM_ARMS):
            x_step = 2 * np.pi * periods[c] / NUM_TRIALS
            eta = NOISE_SD * np.random.randn()
            y[c, t] = offset + AMPLITUDE[c] * np.sin(shift_value[c]) + eta
            shift_value[c] += x_step
    return y


def _permutations(lst):
    return [list(p) for p in itertools.permutations(lst)]


def generate_training_value_functions(offset=50) -> np.ndarray:
    """Training value functions.

    Frequencies and phase shifts are drawn from ranges that exclude those used
    for the participants, so the training and participant (test) value functions
    are disjoint.
    """
    period_ranges = [np.arange(3.1, 3.6, 0.1),
                     np.arange(2.1, 2.6, 0.1),
                     np.arange(6.1, 6.6, 0.1)]
    shift_ranges = [np.arange(0.0, 0.4, 0.1),
                    np.arange(0.4, 0.7, 0.1),
                    np.arange(0.7, 0.9, 0.1)]

    periods = [[a, b, c]
               for a in period_ranges[0]
               for b in period_ranges[1]
               for c in period_ranges[2]]
    shifts = []
    for a in shift_ranges[0]:
        for b in shift_ranges[1]:
            for c in shift_ranges[2]:
                shifts.extend(_permutations([a, b, c]))

    y = np.zeros((len(shifts) * len(periods), NUM_ARMS, NUM_TRIALS))
    i = 0
    for p in periods:
        for s in shifts:
            y[i] = create_sin_value_functions(p, s, offset=offset)
            i += 1
    return np.round(y)


def load_test_value_functions():
    """Load the participants' sine value functions used as the model's test set."""
    sin_data = np.load(paths.SIN_DATA_NPY)
    max_reward = np.load(paths.MAX_REWARD_NPY)
    min_punishment = np.load(paths.MIN_PUNISHMENT_NPY)
    return sin_data, max_reward, min_punishment


# ========================================================================== #
# Training loop (A2C)
# ========================================================================== #
def _run_episode(initial_state, hidden_states, model, env):
    n_trials = CFG["task"]["num_trials"]
    n_actions = CFG["task"]["num_actions"]
    values = torch.zeros(n_trials, dtype=torch.float32, device=device)
    actions = torch.zeros(n_trials, dtype=torch.float32, device=device)
    rewards = torch.zeros(n_trials, dtype=torch.float32, device=device)
    actions_probs = torch.zeros(n_trials, n_actions, dtype=torch.float32, device=device)

    state = initial_state
    for t in range(n_trials):
        actions_log_prob, value_t, (h, c) = model(state, hidden_states)
        hidden_states = (h.detach(), c.detach())
        probs_t = torch.exp(actions_log_prob)
        action = torch.multinomial(probs_t[0, 0], 1)
        state, reward = env.step(action)
        values[t] = value_t[0, 0]
        actions[t] = action
        rewards[t] = reward
        actions_probs[t] = probs_t[0, 0]
    return actions_probs, values, rewards, actions, hidden_states


def _compute_returns(rewards, gamma):
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1]
    for t in reversed(range(rewards.shape[0] - 1)):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns


def _compute_loss(actions_probs, values, returns, actions, beta_critic, beta_entropy):
    advantages = returns - values
    log_probs = torch.log(actions_probs + 1e-7)
    idx = torch.arange(actions.size(0))
    actor_loss = -torch.sum(log_probs[idx, actions.long()] * advantages)
    critic_loss = 0.5 * torch.sum(advantages ** 2)
    entropy = -torch.sum(actions_probs * log_probs)
    total = beta_critic * critic_loss + actor_loss - beta_entropy * entropy
    return total


def train(model, num_episodes, env, name="A2C", save_every=5000, verbose=False):
    """Train the A2C model on the bandit environment for ``num_episodes``."""
    hidden_states = _zero_hidden()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), **CFG["train"]["optimizer_params"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **CFG["train"]["scheduler_params"])

    for epsd in range(num_episodes):
        initial_state, _ = env.reset()
        actions_probs, values, rewards, actions, hidden_states = _run_episode(
            initial_state, hidden_states, model, env
        )
        returns = _compute_returns(rewards, CFG["train"]["gamma"])
        loss = _compute_loss(
            actions_probs, values, returns, actions,
            CFG["train"]["beta_critic"][env.get_dist_idx()],
            CFG["train"]["beta_entropy"][env.get_dist_idx()],
        )
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if CFG["train"]["use_scheduler"]:
            scheduler.step()
        if save_every and epsd and epsd % save_every == 0:
            save_checkpoint(LOGS_DIR / name, model, hidden_states, optimizer, verbose)
        if verbose and epsd % 1000 == 0:
            print(f"Episode {epsd}: loss={loss.item():.3f}")
    return model, hidden_states, optimizer


# ========================================================================== #
# Feeding participant behavior through the frozen model
# ========================================================================== #
def get_state_list(subj_actions, subj_rewards):
    """Convert a participant trajectory into the model's per-trial input states.

    At trial t the model receives the participant's action and reward from
    trial t (so its internal state reflects the participant's history).
    ``subj_actions`` must be 0-indexed arm choices.
    """
    state_list = []
    for t in range(len(subj_actions)):
        t_reward = torch.tensor([subj_rewards[t]], dtype=torch.float32, device=device)
        one_hot = F.one_hot(
            torch.tensor([subj_actions[t]], dtype=torch.long, device=device),
            num_classes=CFG["task"]["num_actions"],
        ).squeeze()
        t_trial = torch.tensor([t], dtype=torch.float32, device=device)
        state_list.append(double_unsqz(torch.cat([t_trial, t_reward, one_hot], 0)))
    return state_list


def feed_subject_episode(trained_model, state_list, hidden_states=None):
    """Run a participant's state sequence through the frozen model.

    Returns per-trial LSTM activations (n_trials x n_units), actor probabilities
    (n_trials x n_actions), and critic value estimates (n_trials,).
    """
    if hidden_states is None:
        hidden_states = _zero_hidden()
    n_units = CFG["net"]["hidden_size"]
    n_actions = CFG["task"]["num_actions"]
    lstm_activations = np.zeros((len(state_list), n_units))
    actions_probs = np.zeros((len(state_list), n_actions))
    critic_values = np.zeros(len(state_list))

    for t, state in enumerate(state_list):
        with torch.no_grad():
            lstm_output, hidden_states = trained_model.lstm(state, hidden_states)
            actions_log_prob = trained_model.actor(lstm_output)
            value_t = trained_model.critic(lstm_output)
        lstm_activations[t] = lstm_output.squeeze().numpy()
        actions_probs[t] = torch.exp(actions_log_prob)[0, 0].numpy()
        critic_values[t] = value_t.item()
    return lstm_activations, actions_probs, critic_values


def calc_likelihood(subj_actions, subj_fb, trained_model, n_iter=10):
    """Overall likelihood: fraction of trials where the model's sampled choice
    matches the participant's *next* choice, averaged over ``n_iter`` samples."""
    state_list = get_state_list(subj_actions, subj_fb)
    _, actions_probs, _ = feed_subject_episode(trained_model, state_list)
    scores = np.zeros(n_iter)
    for n in range(n_iter):
        chosen = np.array(
            [torch.multinomial(torch.tensor(p, dtype=torch.float32), 1).item()
             for p in actions_probs]
        )
        matches = sum(subj_actions[i + 1] == chosen[i] for i in range(len(subj_actions) - 1))
        scores[n] = matches / len(subj_actions)
    return scores.mean()


def feed_all_subjects(trained_model, behavior_gain, behavior_loss, n_subj):
    """Feed every subject's Gain and Loss episode through the frozen model.

    ``behavior_gain[subj]`` / ``behavior_loss[subj]`` are DataFrames with 1-indexed
    ``choice`` and ``FB`` columns. Returns dictionaries keyed by 'Gain'/'Loss',
    each a list (per subject) of LSTM activations, actor probs, and critic values.
    """
    out = {c: {"lstm": [], "probs": [], "critic": []} for c in ("Gain", "Loss")}
    for cond, behavior in (("Gain", behavior_gain), ("Loss", behavior_loss)):
        for subj in range(n_subj):
            actions = behavior[subj].choice.to_numpy() - 1  # -> 0-indexed
            fb = behavior[subj].FB.to_numpy()
            lstm, probs, critic = feed_subject_episode(
                trained_model, get_state_list(actions, fb)
            )
            out[cond]["lstm"].append(lstm)
            out[cond]["probs"].append(probs)
            out[cond]["critic"].append(critic)
    return out
