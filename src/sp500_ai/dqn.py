from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Callable
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
    episodes: int = 500
    batch_size: int = 128
    replay_size: int = 20000
    warmup_steps: int = 1200
    hidden_dim: int = 256
    dropout: float = 0.15
    lr: float = 2e-4
    weight_decay: float = 1e-5
    gamma: float = 0.995
    tau: float = 0.01
    grad_clip: float = 1.0
    target_update_interval: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 12000
    transaction_cost: float = 0.0005
    hold_penalty: float = 0.00005
    overtrade_penalty: float = 0.0001
    reward_scale: float = 100.0
    checkpoint_interval: int = 50
    eval_interval: int = 20
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
        self.reset()

    def reset(self) -> np.ndarray:
        self.idx = 0
        self.position = 0
        self.done = False
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

        reward = new_position * log_return
        reward -= trade_size * self.cfg.transaction_cost
        reward -= abs(new_position) * self.cfg.hold_penalty
        reward -= trade_size * self.cfg.overtrade_penalty
        reward *= self.cfg.reward_scale

        self.position = new_position
        self.idx += 1
        self.done = self.idx >= len(self.states) - 1
        info = {
            "log_return": log_return,
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
    progress_callback: Callable[[dict], None] | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    set_seed(cfg.seed)

    states, close, scaler, feature_columns = _make_states(data_path, cfg.seq_len)
    env = SP500TradingEnv(states, close, cfg)
    state_dim = states.shape[1] + 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DuelingQNetwork(state_dim, cfg.hidden_dim, cfg.dropout).to(device)
    target_net = DuelingQNetwork(state_dim, cfg.hidden_dim, cfg.dropout).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    replay = PrioritizedReplayBuffer(cfg.replay_size)
    beta_start = 0.4
    beta_frames = cfg.episodes * len(states)

    best_eval_reward = float("-inf")
    total_steps = 0
    train_start = time.perf_counter()

    for episode in range(1, cfg.episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_net_log_profit = 0.0
        buy_actions = 0
        sell_actions = 0
        trade_count = 0

        while not done:
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
            ep_net_log_profit += info["position"] * info["log_return"] - info["trade_size"] * cfg.transaction_cost
            trade_count += int(info["trade_size"] > 0)
            if action == 1:
                buy_actions += 1
            elif action == 2:
                sell_actions += 1
            total_steps += 1

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
        eval_reward = None
        summary = (
            f"episode={episode:04d} profit={profit_pct:+.2f}% buys={buy_actions} sells={sell_actions} "
            f"trades={trade_count} train_reward={ep_reward:.4f}"
        )
        if episode % cfg.eval_interval == 0:
            eval_reward = evaluate_policy(env, policy_net, device)
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
                    "train_reward": ep_reward,
                    "profit_pct": profit_pct,
                    "buy_actions": buy_actions,
                    "sell_actions": sell_actions,
                    "trade_count": trade_count,
                    "eval_reward": eval_reward,
                    "epsilon": _epsilon(total_steps, cfg),
                    "progress": episode / max(cfg.episodes, 1),
                    "eta_seconds": max(0.0, eta_seconds),
                    "elapsed_seconds": elapsed,
                }
            )

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
            },
            f,
            indent=2,
        )


def evaluate_policy(env: SP500TradingEnv, policy_net: DuelingQNetwork, device: torch.device) -> float:
    state = env.reset()
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dqn(args.data, args.output, DQNConfig())
