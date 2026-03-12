from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

DEFAULT_DQN_PROFILES: dict[str, dict[str, object]] = {
    "balanced": {
        "description": "Balanced baseline with moderate exploration decay and medium friction.",
        "params": {
            "episodes": 700,
            "epsilon_end": 0.02,
            "epsilon_decay_steps": 140000,
            "transaction_cost": 0.002,
            "hold_penalty": 0.00002,
            "overtrade_penalty": 0.0009,
            "reward_scale": 8.0,
            "min_train_window": 900,
            "max_train_window": 2400,
            "recent_bias_strength": 1.3,
        },
    },
    "conservative": {
        "description": "Lower churn and smoother behavior; stronger trade penalties and wider windows.",
        "params": {
            "episodes": 600,
            "epsilon_end": 0.01,
            "epsilon_decay_steps": 180000,
            "transaction_cost": 0.003,
            "hold_penalty": 0.00003,
            "overtrade_penalty": 0.0015,
            "reward_scale": 6.0,
            "min_train_window": 1200,
            "max_train_window": 2800,
            "recent_bias_strength": 1.1,
        },
    },
    "aggressive": {
        "description": "More reactive profile with faster exploration decay and tighter training windows.",
        "params": {
            "episodes": 900,
            "epsilon_end": 0.03,
            "epsilon_decay_steps": 90000,
            "transaction_cost": 0.0015,
            "hold_penalty": 0.00001,
            "overtrade_penalty": 0.0006,
            "reward_scale": 10.0,
            "min_train_window": 700,
            "max_train_window": 1800,
            "recent_bias_strength": 1.6,
        },
    },
}


def ensure_profile_db(db_path: Path, base_params: Mapping[str, object]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_profiles (
                profile_name TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                params_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        for name, profile in DEFAULT_DQN_PROFILES.items():
            row = conn.execute("SELECT profile_name FROM model_profiles WHERE profile_name = ?", (name,)).fetchone()
            if row:
                continue
            merged = dict(base_params)
            merged.update(profile["params"])
            conn.execute(
                """
                INSERT INTO model_profiles (profile_name, description, params_json)
                VALUES (?, ?, ?)
                """,
                (name, str(profile["description"]), json.dumps(merged)),
            )
        conn.commit()


def list_profiles(db_path: Path) -> list[dict[str, str]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT profile_name, description FROM model_profiles ORDER BY profile_name"
        ).fetchall()
    return [{"name": row[0], "description": row[1]} for row in rows]


def load_profile_params(db_path: Path, profile_name: str) -> dict[str, object] | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT params_json FROM model_profiles WHERE profile_name = ?",
            (profile_name,),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def save_profile(
    db_path: Path,
    profile_name: str,
    description: str,
    params: Mapping[str, object],
) -> None:
    payload = json.dumps(dict(params))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO model_profiles (profile_name, description, params_json, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(profile_name) DO UPDATE SET
                description = excluded.description,
                params_json = excluded.params_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (profile_name, description, payload),
        )
        conn.commit()
