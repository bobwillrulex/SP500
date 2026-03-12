from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Callable
from typing import Any
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data import load_ohlcv_csv
from .features import build_features


@dataclass
class DQNConfig:
    seq_len: int = 30
    episodes: int = 700
    batch_size: int = 256
    replay_size: int = 20000
    warmup_steps: int = 2000
    hidden_dim: int = 256
    dropout: float = 0.15
    lr: float = 2e-4
    weight_decay: float = 1e-5
    gamma: float = 0.99
    tau: float = 0.01
    grad_clip: float = 1.0
    target_update_interval: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 140000
    transaction_cost: float = 0.002
    hold_penalty: float = 0.00002
    overtrade_penalty: float = 0.0009
    reward_scale: float = 8.0
    train_split: float = 0.8
    max_abs_log_return: float = 0.03
    min_train_window: int = 900
    max_train_window: int = 2400
    recent_bias_strength: float = 1.3
    checkpoint_interval: int = 50
    eval_interval: int = 20
    normalize_episode_reward: bool = True
    seed: int = 42


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: tuple[np.ndarray, int, float, np.ndarray, float], priority: float | None = None) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0
        prio = max(priority or max_prio, 1e-6)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> tuple[np.ndarray, list, np.ndarray]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: len(self.buffer)]

        probs = prios**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return indices, samples, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = max(float(prio), 1e-6)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        value = self.value_head(h)
        advantage = self.adv_head(h)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class SP500TradingEnv:
    def __init__(self, states: np.ndarray, close: np.ndarray, cfg: DQNConfig) -> None:
        self.states = states
        self.close = close
        self.cfg = cfg
        self.return_clip_events = 0
        self.reset()

    def reset(self, start_idx: int = 0, end_idx: int | None = None) -> np.ndarray:
        max_end = len(self.states) - 1
        self.start_idx = int(max(0, min(start_idx, max_end - 1)))
        target_end = max_end if end_idx is None else int(end_idx)
        self.end_idx = int(max(self.start_idx + 1, min(target_end, max_end)))
        self.idx = self.start_idx
        self.position = 0
        self.done = False
        self.return_clip_events = 0
        return self._state()

    def _state(self) -> np.ndarray:
        base = self.states[self.idx]
        pos_features = np.array([self.position, self.idx / max(len(self.states) - 1, 1)], dtype=np.float32)
        return np.concatenate([base, pos_features], axis=0).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        new_position = {0: self.position, 1: 1, 2: -1}[action]
        trade_size = abs(new_position - self.position)

        price_now = self.close[self.idx]
        price_next = self.close[self.idx + 1]
        log_return = float(math.log((price_next + 1e-9) / (price_now + 1e-9)))
        clipped_log_return = float(np.clip(log_return, -self.cfg.max_abs_log_return, self.cfg.max_abs_log_return))
        if not math.isclose(clipped_log_return, log_return):
            self.return_clip_events += 1

        reward = new_position * clipped_log_return
        reward -= trade_size * self.cfg.transaction_cost
        reward -= abs(new_position) * self.cfg.hold_penalty
        reward -= trade_size * self.cfg.overtrade_penalty
        reward *= self.cfg.reward_scale

        self.position = new_position
        self.idx += 1
        self.done = self.idx >= self.end_idx
        info = {
            "log_return": log_return,
            "clipped_log_return": clipped_log_return,
            "trade_size": float(trade_size),
            "position": float(new_position),
        }
        return self._state(), float(reward), self.done, info


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_states(data_path: str, seq_len: int) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str]]:
    df = load_ohlcv_csv(data_path)
    feat = build_features(df)
    aligned = df.loc[feat.index].copy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feat.values).astype(np.float32)

    states: list[np.ndarray] = []
    closes: list[float] = []
    for i in range(seq_len, len(x_scaled)):
        states.append(x_scaled[i - seq_len : i].reshape(-1))
        closes.append(float(aligned.iloc[i]["close"]))

    states_arr = np.asarray(states, dtype=np.float32)
    close_arr = np.asarray(closes, dtype=np.float32)
    if len(states_arr) < 200:
        raise ValueError("Not enough data to train DQN. Need more daily candles.")
    return states_arr, close_arr, scaler, list(feat.columns)


def _split_train_eval(
    states: np.ndarray,
    close: np.ndarray,
    train_split: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_idx = int(len(states) * train_split)
    split_idx = min(max(split_idx, 200), len(states) - 200)
    train_states = states[:split_idx]
    train_close = close[:split_idx]
    eval_states = states[split_idx:]
    eval_close = close[split_idx:]
    if len(train_states) < 200 or len(eval_states) < 200:
        raise ValueError("Not enough samples to create train/eval split. Increase data history or adjust train_split.")
    return train_states, train_close, eval_states, eval_close


def _sample_episode_slice(total_len: int, cfg: DQNConfig) -> tuple[int, int]:
    max_start = max(total_len - 2, 1)
    if total_len <= cfg.min_train_window + 1:
        return 0, max_start

    max_window = min(cfg.max_train_window, total_len - 1)
    min_window = min(cfg.min_train_window, max_window)
    window = random.randint(min_window, max_window)

    # Bias episode windows toward recent history while still sampling older data.
    u = random.random()
    recency = 1.0 - (1.0 - u) ** max(cfg.recent_bias_strength, 1.0)
    end_idx = int(recency * (total_len - 1))
    end_idx = max(window, min(end_idx, total_len - 1))
    start_idx = max(0, end_idx - window)
    return start_idx, end_idx


def _epsilon(step: int, cfg: DQNConfig) -> float:
    ratio = min(step / max(cfg.epsilon_decay_steps, 1), 1.0)
    return cfg.epsilon_start + ratio * (cfg.epsilon_end - cfg.epsilon_start)


def _save_checkpoint(
    output_dir: str,
    name: str,
    policy_net: DuelingQNetwork,
    target_net: DuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    episode: int,
    steps: int,
    best_reward: float,
) -> None:
    torch.save(
        {
            "policy": policy_net.state_dict(),
            "target": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "episode": episode,
            "steps": steps,
            "best_eval_reward": best_reward,
        },
        os.path.join(output_dir, name),
    )


def train_dqn(
    data_path: str,
    output_dir: str,
    cfg: DQNConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    set_seed(cfg.seed)

    states, close, scaler, feature_columns = _make_states(data_path, cfg.seq_len)
    train_states, train_close, eval_states, eval_close = _split_train_eval(states, close, cfg.train_split)
    env = SP500TradingEnv(train_states, train_close, cfg)
    eval_env = SP500TradingEnv(eval_states, eval_close, cfg)
    state_dim = states.shape[1] + 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DuelingQNetwork(state_dim, cfg.hidden_dim, cfg.dropout).to(device)
    target_net = DuelingQNetwork(state_dim, cfg.hidden_dim, cfg.dropout).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    replay = PrioritizedReplayBuffer(cfg.replay_size)
    beta_start = 0.4
    beta_frames = cfg.episodes * len(train_states)

    best_eval_reward = float("-inf")
    total_steps = 0
    train_start = time.perf_counter()

    for episode in range(1, cfg.episodes + 1):
        slice_start, slice_end = _sample_episode_slice(len(train_states), cfg)
        state = env.reset(start_idx=slice_start, end_idx=slice_end)
        done = False
        ep_reward = 0.0
        ep_net_log_profit = 0.0
        episode_steps = 0
        buy_actions = 0
        sell_actions = 0
        trade_count = 0

        while not done:
            if stop_requested is not None and stop_requested():
                print("Early stop requested; ending DQN training loop.")
                done = True
                break
            eps = _epsilon(total_steps, cfg)
            if random.random() < eps:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(policy_net(st).argmax(dim=1).item())

            next_state, reward, done, info = env.step(action)
            replay.add((state, action, reward, next_state, float(done)))
            state = next_state
            ep_reward += reward
            ep_net_log_profit += info["position"] * info["clipped_log_return"] - info["trade_size"] * cfg.transaction_cost
            trade_count += int(info["trade_size"] > 0)
            if action == 1:
                buy_actions += 1
            elif action == 2:
                sell_actions += 1
            total_steps += 1
            episode_steps += 1

            if len(replay) < max(cfg.batch_size, cfg.warmup_steps):
                continue

            beta = min(1.0, beta_start + (1.0 - beta_start) * (total_steps / max(beta_frames, 1)))
            indices, samples, weights = replay.sample(cfg.batch_size, beta)

            s = torch.tensor(np.array([x[0] for x in samples]), dtype=torch.float32, device=device)
            a = torch.tensor([x[1] for x in samples], dtype=torch.long, device=device).unsqueeze(1)
            r = torch.tensor([x[2] for x in samples], dtype=torch.float32, device=device).unsqueeze(1)
            ns = torch.tensor(np.array([x[3] for x in samples]), dtype=torch.float32, device=device)
            d = torch.tensor([x[4] for x in samples], dtype=torch.float32, device=device).unsqueeze(1)
            w = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)

            q_vals = policy_net(s).gather(1, a)
            with torch.no_grad():
                next_actions = policy_net(ns).argmax(dim=1, keepdim=True)
                next_q = target_net(ns).gather(1, next_actions)
                target = r + cfg.gamma * (1 - d) * next_q

            td_error = target - q_vals
            loss = (w * nn.functional.smooth_l1_loss(q_vals, target, reduction="none")).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.grad_clip)
            optimizer.step()

            replay.update_priorities(indices, td_error.detach().abs().cpu().numpy().flatten() + 1e-6)

            if total_steps % cfg.target_update_interval == 0:
                for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(cfg.tau * param.data + (1.0 - cfg.tau) * target_param.data)

        profit_pct = (math.exp(ep_net_log_profit) - 1.0) * 100.0
        reward_denom = max(episode_steps, 1) if cfg.normalize_episode_reward else 1
        normalized_ep_reward = ep_reward / reward_denom
        eval_reward = None
        summary = (
            f"episode={episode:04d} profit={profit_pct:+.2f}% buys={buy_actions} sells={sell_actions} "
            f"trades={trade_count} train_reward={normalized_ep_reward:.4f} raw_reward={ep_reward:.4f} steps={episode_steps} return_clips={env.return_clip_events} "
            f"slice={slice_start}:{slice_end}"
        )
        if episode % cfg.eval_interval == 0:
            eval_reward = evaluate_policy(eval_env, policy_net, device)
            print(f"{summary} eval_reward={eval_reward:.4f} eps={_epsilon(total_steps, cfg):.4f}")
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(policy_net.state_dict(), os.path.join(output_dir, "best_dqn_policy.pt"))
        else:
            print(f"{summary} eps={_epsilon(total_steps, cfg):.4f}")

        elapsed = time.perf_counter() - train_start
        eta_seconds = (elapsed / episode) * (cfg.episodes - episode) if episode else 0.0
        if progress_callback is not None:
            progress_callback(
                {
                    "kind": "dqn",
                    "episode": episode,
                    "total_episodes": cfg.episodes,
                    "train_reward": normalized_ep_reward,
                    "train_reward_raw": ep_reward,
                    "episode_steps": episode_steps,
                    "profit_pct": profit_pct,
                    "buy_actions": buy_actions,
                    "sell_actions": sell_actions,
                    "trade_count": trade_count,
                    "return_clip_events": env.return_clip_events,
                    "eval_reward": eval_reward,
                    "slice_start": slice_start,
                    "slice_end": slice_end,
                    "epsilon": _epsilon(total_steps, cfg),
                    "progress": episode / max(cfg.episodes, 1),
                    "eta_seconds": max(0.0, eta_seconds),
                    "elapsed_seconds": elapsed,
                }
            )

        if stop_requested is not None and stop_requested():
            break

        if episode % cfg.checkpoint_interval == 0:
            _save_checkpoint(
                output_dir,
                f"checkpoint_ep{episode:04d}.pt",
                policy_net,
                target_net,
                optimizer,
                episode,
                total_steps,
                best_eval_reward,
            )

    torch.save(policy_net.state_dict(), os.path.join(output_dir, "last_dqn_policy.pt"))
    joblib.dump(scaler, os.path.join(output_dir, "dqn_scaler.pkl"))

    with open(os.path.join(output_dir, "dqn_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "feature_columns": feature_columns,
                "state_dim": state_dim,
                "best_eval_reward": best_eval_reward,
                "device": str(device),
                "num_samples": int(len(states)),
                "train_samples": int(len(train_states)),
                "eval_samples": int(len(eval_states)),
            },
            f,
            indent=2,
        )


def evaluate_policy(env: SP500TradingEnv, policy_net: DuelingQNetwork, device: torch.device) -> float:
    state = env.reset(start_idx=0, end_idx=len(env.states) - 1)
    done = False
    reward = 0.0
    while not done:
        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(policy_net(st).argmax(dim=1).item())
        state, r, done, _ = env.step(action)
        reward += r
    return reward


def predict_dqn_action(data_path: str, model_path: str, scaler_path: str, meta_path: str) -> str:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg = DQNConfig(**meta["config"])
    scaler = joblib.load(scaler_path)

    df = load_ohlcv_csv(data_path)
    feat = build_features(df)
    x_scaled = scaler.transform(feat.values).astype(np.float32)
    window = x_scaled[-cfg.seq_len :].reshape(1, -1)
    state = np.concatenate([window[0], np.array([0.0, 1.0], dtype=np.float32)], axis=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DuelingQNetwork(meta["state_dim"], cfg.hidden_dim, cfg.dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        q_values = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))[0]
        action = int(torch.argmax(q_values).item())

    mapping = {0: "HOLD", 1: "BUY", 2: "SELL"}
    return mapping[action]


def _build_config_from_args(args: argparse.Namespace) -> DQNConfig:
    cfg_values = asdict(DQNConfig())

    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for key, value in loaded.items():
            if key in cfg_values:
                cfg_values[key] = value

    for key in cfg_values:
        override = getattr(args, key, None)
        if override is not None:
            cfg_values[key] = override

    return DQNConfig(**cfg_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional path to a JSON file containing DQNConfig field overrides.",
    )

    defaults = asdict(DQNConfig())
    for key, value in defaults.items():
        value_type = type(value)
        if isinstance(value, bool):
            continue
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key, type=value_type, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dqn(args.data, args.output, _build_config_from_args(args))
