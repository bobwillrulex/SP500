from __future__ import annotations

import datetime as dt
import json
import os
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from tkinter import StringVar, Tk, ttk

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sp500_ai.config import TrainConfig
from sp500_ai.dqn import DQNConfig, predict_dqn_action, train_dqn
from sp500_ai.predict import predict_next_close
from sp500_ai.train import train_once
from sp500_ai.yahoo import fetch_sp500_history

SETTINGS_PATH = ROOT / ".gui_settings.json"


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


class SP500AIGUI(Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("S&P 500 AI Studio")
        self.geometry("1280x860")
        self.minsize(1120, 780)

        self._event_queue: queue.Queue[dict] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._continuous_stop = threading.Event()
        self._continuous_running = False

        self.defaults = {
            "yahoo": {
                "data_path": "data/sp500.csv",
                "db_path": "data/sp500.db",
                "period": "max",
            },
            "forecast": {
                "output_dir": "artifacts/run1",
                "model_path": "artifacts/run1/best_model.pt",
                "scaler_path": "artifacts/run1/scaler.pkl",
                "meta_path": "artifacts/run1/meta.json",
                "continuous_output": "artifacts/live",
                "interval_seconds": "3600",
            },
            "dqn": {
                "output_dir": "artifacts/dqn_run1",
                "model_path": "artifacts/dqn_run1/best_dqn_policy.pt",
                "scaler_path": "artifacts/dqn_run1/dqn_scaler.pkl",
                "meta_path": "artifacts/dqn_run1/dqn_meta.json",
            },
            "forecast_params": {k: str(v) for k, v in asdict(TrainConfig()).items()},
            "dqn_params": {k: str(v) for k, v in asdict(DQNConfig()).items()},
        }

        self.settings = self._load_settings()
        self.vars: dict[str, StringVar] = {}
        self._init_style()
        self._build_ui()
        self.after(120, self._drain_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        bg = "#0f172a"
        panel = "#111827"
        text = "#e5e7eb"
        accent = "#22d3ee"
        muted = "#94a3b8"
        self.configure(bg=bg)

        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background="#1f2937", foreground=text, padding=(16, 8))
        style.map("TNotebook.Tab", background=[("selected", "#334155")])
        style.configure("Card.TFrame", background=panel)
        style.configure("Header.TLabel", background=bg, foreground=text, font=("Segoe UI", 18, "bold"))
        style.configure("Subheader.TLabel", background=panel, foreground=text, font=("Segoe UI", 12, "bold"))
        style.configure("Field.TLabel", background=panel, foreground=muted, font=("Segoe UI", 10))
        style.configure("Output.TLabel", background=panel, foreground=accent, font=("Segoe UI", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 6))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7))
        style.map("Accent.TButton", background=[("!disabled", "#0891b2")], foreground=[("!disabled", "white")])

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="Card.TFrame", padding=18)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="S&P 500 AI Studio", style="Header.TLabel").pack(anchor="w", pady=(0, 10))

        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=True)

        forecast_tab = ttk.Frame(tabs, style="Card.TFrame", padding=14)
        dqn_tab = ttk.Frame(tabs, style="Card.TFrame", padding=14)
        tabs.add(forecast_tab, text="Forecast Pipeline")
        tabs.add(dqn_tab, text="DQN Trading Bot")

        self._build_shared_panel(forecast_tab)
        self._build_forecast_panel(forecast_tab)

        self._build_shared_panel(dqn_tab)
        self._build_dqn_panel(dqn_tab)

    def _var(self, key: str, default: str, legacy_keys: tuple[str, ...] = ()) -> StringVar:
        value = self.settings.get(key)
        if value is None:
            for legacy in legacy_keys:
                if legacy in self.settings:
                    value = self.settings[legacy]
                    break
        if value is None:
            value = default
        var = StringVar(value=value)
        self.vars[key] = var
        return var

    def _add_field(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
        default: str,
        legacy_keys: tuple[str, ...] = (),
    ) -> None:
        ttk.Label(parent, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(parent, textvariable=self._var(key, default, legacy_keys), width=40).grid(row=row, column=1, sticky="ew", pady=4)

    def _build_shared_panel(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill="x", pady=(0, 10))
        card.columnconfigure(1, weight=1)
        ttk.Label(card, text="Market Data", style="Subheader.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self._add_field(card, 1, "CSV path", "yahoo.data_path", self.defaults["yahoo"]["data_path"], ("shared.data_path",))
        self._add_field(card, 2, "SQLite path (blank to skip)", "yahoo.db_path", self.defaults["yahoo"]["db_path"], ("shared.db_path",))
        self._add_field(card, 3, "Yahoo period", "yahoo.period", self.defaults["yahoo"]["period"], ("shared.period",))

    def _build_forecast_panel(self, parent: ttk.Frame) -> None:
        actions = ttk.Frame(parent, style="Card.TFrame")
        actions.pack(fill="x", pady=(0, 10))
        ttk.Label(actions, text="Actions", style="Subheader.TLabel").pack(anchor="w", pady=(0, 8))

        row = ttk.Frame(actions, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="Download Data", style="Accent.TButton", command=self.download_data).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Train Forecast", style="Accent.TButton", command=self.train_forecast).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Predict Next Close", command=self.predict_forecast).pack(side="left")

        io_card = ttk.Frame(parent, style="Card.TFrame")
        io_card.pack(fill="x", pady=(0, 10))
        io_card.columnconfigure(1, weight=1)
        ttk.Label(io_card, text="Forecast Artifacts", style="Subheader.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._add_field(io_card, 1, "Output dir", "forecast.output_dir", self.defaults["forecast"]["output_dir"])
        self._add_field(io_card, 2, "Model path", "forecast.model_path", self.defaults["forecast"]["model_path"])
        self._add_field(io_card, 3, "Scaler path", "forecast.scaler_path", self.defaults["forecast"]["scaler_path"])
        self._add_field(io_card, 4, "Meta path", "forecast.meta_path", self.defaults["forecast"]["meta_path"])

        cont_row = ttk.Frame(io_card, style="Card.TFrame")
        cont_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        cont_row.columnconfigure(1, weight=1)
        ttk.Label(cont_row, text="Continuous output", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(cont_row, textvariable=self._var("forecast.continuous_output", self.defaults["forecast"]["continuous_output"])).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Label(cont_row, text="Interval seconds", style="Field.TLabel").grid(row=0, column=2, padx=(8, 0))
        ttk.Entry(cont_row, width=10, textvariable=self._var("forecast.interval_seconds", self.defaults["forecast"]["interval_seconds"])).grid(row=0, column=3, padx=(8, 0))
        self.forecast_cont_btn = ttk.Button(cont_row, text="Start Continuous", command=self.toggle_continuous)
        self.forecast_cont_btn.grid(row=0, column=4, padx=(10, 0))

        params_card = self._build_param_grid(parent, "Forecast Parameters", "forecast_params", self.defaults["forecast_params"], self.reset_forecast_defaults)

        self.forecast_progress = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self.forecast_progress.pack(fill="x", pady=(8, 4))
        self.forecast_status = ttk.Label(parent, text="Idle", style="Output.TLabel")
        self.forecast_status.pack(anchor="w")

        self.forecast_log = self._build_log(parent)

    def _build_dqn_panel(self, parent: ttk.Frame) -> None:
        actions = ttk.Frame(parent, style="Card.TFrame")
        actions.pack(fill="x", pady=(0, 10))
        ttk.Label(actions, text="Actions", style="Subheader.TLabel").pack(anchor="w", pady=(0, 8))

        row = ttk.Frame(actions, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="Download Data", style="Accent.TButton", command=self.download_data).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Train DQN", style="Accent.TButton", command=self.train_dqn_ui).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="Predict Action", command=self.predict_dqn_ui).pack(side="left")

        io_card = ttk.Frame(parent, style="Card.TFrame")
        io_card.pack(fill="x", pady=(0, 10))
        io_card.columnconfigure(1, weight=1)
        ttk.Label(io_card, text="DQN Artifacts", style="Subheader.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._add_field(io_card, 1, "Output dir", "dqn.output_dir", self.defaults["dqn"]["output_dir"])
        self._add_field(io_card, 2, "Model path", "dqn.model_path", self.defaults["dqn"]["model_path"])
        self._add_field(io_card, 3, "Scaler path", "dqn.scaler_path", self.defaults["dqn"]["scaler_path"])
        self._add_field(io_card, 4, "Meta path", "dqn.meta_path", self.defaults["dqn"]["meta_path"])

        self._build_param_grid(parent, "DQN Parameters", "dqn_params", self.defaults["dqn_params"], self.reset_dqn_defaults)

        self.dqn_progress = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self.dqn_progress.pack(fill="x", pady=(8, 4))
        self.dqn_status = ttk.Label(parent, text="Idle", style="Output.TLabel")
        self.dqn_status.pack(anchor="w")

        self.dqn_log = self._build_log(parent)

    def _build_param_grid(self, parent: ttk.Frame, title: str, prefix: str, defaults: dict[str, str], reset_command) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill="x", pady=(0, 8))
        header = ttk.Frame(card, style="Card.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text=title, style="Subheader.TLabel").pack(side="left", pady=(0, 8))
        ttk.Button(header, text="Reset to Defaults", command=reset_command).pack(side="right")

        grid = ttk.Frame(card, style="Card.TFrame")
        grid.pack(fill="x")

        items = list(defaults.items())
        per_col = (len(items) + 1) // 2
        for col in range(2):
            grid.columnconfigure(col * 2 + 1, weight=1)

        for idx, (name, default) in enumerate(items):
            col = idx // per_col
            row = idx % per_col
            base_col = col * 2
            ttk.Label(grid, text=name, style="Field.TLabel").grid(row=row, column=base_col, sticky="w", padx=(0, 8), pady=3)
            ttk.Entry(grid, width=16, textvariable=self._var(f"{prefix}.{name}", default)).grid(row=row, column=base_col + 1, sticky="ew", padx=(0, 20), pady=3)
        return card

    def _build_log(self, parent: ttk.Frame):
        text = __import__("tkinter").Text(parent, height=8, bg="#020617", fg="#cbd5e1", insertbackground="white")
        text.pack(fill="both", expand=True, pady=(6, 2))
        return text

    def _load_settings(self) -> dict[str, str]:
        if SETTINGS_PATH.exists():
            try:
                return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_settings(self) -> None:
        payload = {k: v.get() for k, v in self.vars.items()}
        SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def reset_forecast_defaults(self) -> None:
        for k, v in self.defaults["forecast_params"].items():
            self.vars[f"forecast_params.{k}"].set(v)
        self.log(self.forecast_log, "Forecast params reset to defaults")

    def reset_dqn_defaults(self) -> None:
        for k, v in self.defaults["dqn_params"].items():
            self.vars[f"dqn_params.{k}"].set(v)
        self.log(self.dqn_log, "DQN params reset to defaults")

    def _forecast_cfg(self) -> TrainConfig:
        payload = asdict(TrainConfig())
        for k, default in payload.items():
            raw = self.vars[f"forecast_params.{k}"].get()
            payload[k] = self._cast_type(raw, default)
        return TrainConfig(**payload)

    def _dqn_cfg(self) -> DQNConfig:
        payload = asdict(DQNConfig())
        for k, default in payload.items():
            raw = self.vars[f"dqn_params.{k}"].get()
            payload[k] = self._cast_type(raw, default)
        return DQNConfig(**payload)

    @staticmethod
    def _cast_type(raw: str, default):
        if isinstance(default, bool):
            return raw.lower() in {"1", "true", "yes", "on"}
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        return raw

    def _start_worker(self, task_name: str, fn) -> None:
        def runner() -> None:
            try:
                fn()
                self._event_queue.put({"type": "done", "task": task_name})
            except Exception as exc:
                self._event_queue.put({"type": "error", "task": task_name, "error": str(exc)})

        t = threading.Thread(target=runner, daemon=True)
        self._workers.append(t)
        t.start()

    def log(self, widget, message: str) -> None:
        stamp = dt.datetime.utcnow().strftime("%H:%M:%S")
        widget.insert("end", f"[{stamp}] {message}\n")
        widget.see("end")

    def download_data(self) -> None:
        self._save_settings()

        def job() -> None:
            fetch_sp500_history(
                output_path=self.vars["yahoo.data_path"].get(),
                period=self.vars["yahoo.period"].get(),
                db_path=self.vars["yahoo.db_path"].get().strip() or None,
            )

        self._start_worker("download", job)

    def train_forecast(self) -> None:
        self._save_settings()
        cfg = self._forecast_cfg()
        self.forecast_progress["value"] = 0
        self.forecast_status.configure(text="Training started...")

        def callback(data: dict) -> None:
            self._event_queue.put({"type": "forecast_progress", "data": data})

        def job() -> None:
            train_once(self.vars["yahoo.data_path"].get(), self.vars["forecast.output_dir"].get(), cfg, progress_callback=callback)

        self._start_worker("forecast_train", job)

    def train_dqn_ui(self) -> None:
        self._save_settings()
        cfg = self._dqn_cfg()
        self.dqn_progress["value"] = 0
        self.dqn_status.configure(text="Training started...")

        def callback(data: dict) -> None:
            self._event_queue.put({"type": "dqn_progress", "data": data})

        def job() -> None:
            train_dqn(self.vars["yahoo.data_path"].get(), self.vars["dqn.output_dir"].get(), cfg, progress_callback=callback)

        self._start_worker("dqn_train", job)

    def predict_forecast(self) -> None:
        self._save_settings()

        def job() -> None:
            pred = predict_next_close(
                self.vars["yahoo.data_path"].get(),
                self.vars["forecast.model_path"].get(),
                self.vars["forecast.scaler_path"].get(),
                self.vars["forecast.meta_path"].get(),
            )
            self._event_queue.put({"type": "forecast_prediction", "value": pred})

        self._start_worker("forecast_predict", job)

    def predict_dqn_ui(self) -> None:
        self._save_settings()

        def job() -> None:
            signal = predict_dqn_action(
                self.vars["yahoo.data_path"].get(),
                self.vars["dqn.model_path"].get(),
                self.vars["dqn.scaler_path"].get(),
                self.vars["dqn.meta_path"].get(),
            )
            self._event_queue.put({"type": "dqn_prediction", "value": signal})

        self._start_worker("dqn_predict", job)

    def toggle_continuous(self) -> None:
        self._save_settings()
        if self._continuous_running:
            self._continuous_stop.set()
            self.forecast_cont_btn.configure(text="Start Continuous")
            self._continuous_running = False
            self.log(self.forecast_log, "Stopping continuous training...")
            return

        self._continuous_stop.clear()
        self._continuous_running = True
        self.forecast_cont_btn.configure(text="Stop Continuous")
        interval = int(self.vars["forecast.interval_seconds"].get())
        output_base = self.vars["forecast.continuous_output"].get()

        def job() -> None:
            while not self._continuous_stop.is_set():
                ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                run_dir = os.path.join(output_base, f"run_{ts}")
                cfg = self._forecast_cfg()
                self._event_queue.put({"type": "log", "target": "forecast", "message": f"continuous training -> {run_dir}"})
                train_once(self.vars["yahoo.data_path"].get(), run_dir, cfg, progress_callback=lambda data: self._event_queue.put({"type": "forecast_progress", "data": data}))
                end = time.time() + interval
                while time.time() < end and not self._continuous_stop.is_set():
                    time.sleep(0.2)

        self._start_worker("continuous", job)

    def _drain_queue(self) -> None:
        while True:
            try:
                evt = self._event_queue.get_nowait()
            except queue.Empty:
                break

            typ = evt["type"]
            if typ == "forecast_progress":
                data = evt["data"]
                self.forecast_progress["value"] = data["progress"] * 100
                self.forecast_status.configure(
                    text=(
                        f"Epoch {data['epoch']}/{data['total_epochs']} | "
                        f"val={data['val_loss']:.5f} | ETA {format_eta(data['eta_seconds'])}"
                    )
                )
            elif typ == "dqn_progress":
                data = evt["data"]
                self.dqn_progress["value"] = data["progress"] * 100
                eval_part = "" if data["eval_reward"] is None else f" eval={data['eval_reward']:.4f}"
                self.dqn_status.configure(
                    text=(
                        f"Episode {data['episode']}/{data['total_episodes']} | "
                        f"reward={data['train_reward']:.4f}{eval_part} | ETA {format_eta(data['eta_seconds'])}"
                    )
                )
            elif typ == "forecast_prediction":
                self.log(self.forecast_log, f"Predicted next close: {evt['value']:.4f}")
            elif typ == "dqn_prediction":
                self.log(self.dqn_log, f"DQN action signal: {evt['value']}")
            elif typ == "log":
                target = self.forecast_log if evt.get("target") == "forecast" else self.dqn_log
                self.log(target, evt["message"])
            elif typ == "done":
                task = evt["task"]
                if task == "download":
                    self.log(self.forecast_log, "Downloaded latest Yahoo historical data")
                    self.log(self.dqn_log, "Downloaded latest Yahoo historical data")
                elif task == "forecast_train":
                    self.log(self.forecast_log, f"Forecast training complete -> {self.vars['forecast.output_dir'].get()}")
                elif task == "dqn_train":
                    self.log(self.dqn_log, f"DQN training complete -> {self.vars['dqn.output_dir'].get()}")
                elif task == "continuous" and self._continuous_running:
                    self._continuous_running = False
                    self.forecast_cont_btn.configure(text="Start Continuous")
            elif typ == "error":
                msg = f"{evt['task']} failed: {evt['error']}"
                self.log(self.forecast_log, msg)
                self.log(self.dqn_log, msg)

        self.after(120, self._drain_queue)

    def _on_close(self) -> None:
        self._continuous_stop.set()
        self._save_settings()
        self.destroy()


def main() -> None:
    app = SP500AIGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
