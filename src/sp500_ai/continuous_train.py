from __future__ import annotations

import argparse
import datetime as dt
import os
import time

from .config import TrainConfig
from .train import train_once


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--interval-seconds", type=int, default=3600)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    while True:
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output, f"run_{ts}")
        print(f"[{dt.datetime.utcnow().isoformat()}] starting training -> {run_dir}")

        try:
            train_once(args.data, run_dir, TrainConfig())
            latest = os.path.join(args.output, "latest")
            if os.path.islink(latest) or os.path.exists(latest):
                try:
                    os.remove(latest)
                except OSError:
                    pass
            os.symlink(run_dir, latest)
            print(f"[{dt.datetime.utcnow().isoformat()}] training finished")
        except Exception as exc:
            print(f"[{dt.datetime.utcnow().isoformat()}] training failed: {exc}")

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
