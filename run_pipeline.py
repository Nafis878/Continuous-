#!/usr/bin/env python3
"""
Run the whole pipeline with one command.

Stages (in order)
-----------------
  1. env        - Python / torch / GPU / dependency check (optionally pip install)
  2. smoke      - Continuous RUL Benchmark quick test: every model, 1 epoch (fails fast)
  3. benchmark  - full Continuous RUL Benchmark (CRB): all datasets x models x seeds
  4. studies    - optional extra experiments:
                    --with-studies   condition clustering (FD002/FD004) + N-CMAPSS multi-file
                    --with-scania    SCANIA classification (family comparison + cost tuning)
                    --with-ablation  LSTM one-factor ablations (uses SCANIA data too)
                  SCANIA data (Git LFS, ~1.7 GB) is pulled automatically when needed.
  5. summary    - status of every stage -> OUT/pipeline_status.json

Everything is written under --out (default: ml_pipeline/results/pipeline):
  OUT/benchmark/   LEADERBOARD.md, runs.csv, leaderboard*.csv   (resumable)
  OUT/studies/     results of the optional studies
  OUT/logs/        one log file per stage
Re-running the same command resumes the benchmark (finished runs are skipped).

Examples
--------
  python run_pipeline.py                                   # env + smoke + full benchmark
  python run_pipeline.py --quick                           # ~2 min check of everything
  python run_pipeline.py --out /content/drive/MyDrive/CRB  # results on a mounted Google Drive
  python run_pipeline.py --models CATT+TCT CATT TCN --datasets FD001 FD004
  python run_pipeline.py --with-studies --with-scania      # everything
  python run_pipeline.py --dry-run --with-scania           # print the plan only
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
ML = REPO / "ml_pipeline"
SCANIA_DATA = REPO / "SCANIA-Component-X" / "data"
SCANIA_LFS = ["train_operational_readouts.csv", "validation_operational_readouts.csv", "test_operational_readouts.csv"]

for _s in (sys.stdout, sys.stderr):  # never crash on console encoding
    try:
        _s.reconfigure(errors="replace")
    except AttributeError:
        pass


def banner(text: str):
    print("\n" + "=" * 78 + f"\n  {text}\n" + "=" * 78, flush=True)


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out).resolve()
        self.logs = self.out / "logs"
        self.status: list[dict] = []
        self.py = sys.executable

    # ------------------------------------------------------------------ #
    def run(self, name: str, cmd: list[str], env: dict | None = None) -> bool:
        """Run a command, stream its output live and tee it to OUT/logs/<name>.log."""
        print(f"$ {' '.join(cmd)}", flush=True)
        if self.args.dry_run:
            self.status.append({"stage": name, "status": "dry-run", "cmd": cmd})
            return True
        self.logs.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        full_env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}
        with open(self.logs / f"{name}.log", "w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen(cmd, cwd=REPO, env=full_env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            rc = proc.wait()
        ok = rc == 0
        self.status.append({"stage": name, "status": "ok" if ok else f"failed (exit {rc})",
                            "minutes": round((time.time() - t0) / 60, 2), "cmd": cmd})
        print(f"--> {name}: {'OK' if ok else f'FAILED (exit {rc})'} in {(time.time() - t0) / 60:.1f} min", flush=True)
        return ok

    def skip(self, name: str, why: str):
        print(f"--> {name}: skipped ({why})", flush=True)
        self.status.append({"stage": name, "status": f"skipped: {why}"})

    # ------------------------------------------------------------------ #
    def stage_env(self) -> bool:
        banner("1/5  Environment")
        if self.args.install and not self.args.dry_run:
            if not self.run("pip_install", [self.py, "-m", "pip", "install", "-q", "-r", str(ML / "requirements.txt")]):
                return False
        print(f"python  {platform.python_version()} ({sys.executable})")
        print(f"repo    {REPO}")
        print(f"output  {self.out}")
        missing = []
        for mod in ("numpy", "pandas", "sklearn", "h5py", "torch"):
            try:
                m = __import__(mod)
                print(f"{mod:8s}{getattr(m, '__version__', '?')}")
            except ImportError:
                missing.append(mod)
        if missing:
            print(f"Missing packages: {missing} -> run with --install or: pip install -r ml_pipeline/requirements.txt")
            self.status.append({"stage": "env", "status": f"failed: missing {missing}"})
            return False
        import torch

        if torch.cuda.is_available():
            print(f"GPU     {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        else:
            print("GPU     none -> running on CPU (much slower; on Colab: Runtime -> Change runtime type -> T4 GPU)")
        nc = os.environ.get("NCMAPSS_DIR")
        print(f"N-CMAPSS dir: {nc or 'N-CMAPSS/data_set (default)'}  (datasets skipped if no .h5 files)")
        self.status.append({"stage": "env", "status": "ok", "gpu": torch.cuda.is_available()})
        return True

    def bench_cmd(self, out: Path, quick: bool) -> list[str]:
        cmd = [self.py, str(ML / "run_benchmark.py"), "--out", str(out)]
        if quick:
            cmd += ["--quick", "--fresh"]
        else:
            cmd += ["--seeds", *map(str, self.args.seeds)]
            if self.args.epochs:
                cmd += ["--epochs", str(self.args.epochs)]
        if self.args.datasets:
            cmd += ["--datasets", *self.args.datasets]
        if self.args.models:
            cmd += ["--models", *self.args.models]
        return cmd

    def stage_smoke(self) -> bool:
        banner("2/5  Smoke test (every model, 1 epoch, small subsets)")
        if self.args.skip_smoke and not self.args.quick:
            self.skip("smoke", "--skip-smoke")
            return True
        return self.run("smoke", self.bench_cmd(self.out / "smoke", quick=True))

    def stage_benchmark(self) -> bool:
        banner("3/5  Continuous RUL Benchmark (CRB)")
        if self.args.quick:
            self.skip("benchmark", "--quick (smoke test only)")
            return True
        return self.run("benchmark", self.bench_cmd(self.out / "benchmark", quick=False))

    # ------------------------------------------------------------------ #
    def scania_data_ready(self) -> bool:
        for f in SCANIA_LFS:
            p = SCANIA_DATA / f
            if not p.exists():
                return False
            with open(p, "rb") as fh:
                if fh.read(40).startswith(b"version https://git-lfs"):
                    return False  # still an LFS pointer file
        return True

    def ensure_scania(self) -> bool:
        if self.scania_data_ready():
            print("SCANIA data present.")
            return True
        print("SCANIA operational CSVs are Git LFS pointers -> downloading (~1.7 GB) ...")
        if shutil.which("git") is None:
            print("git not found; cannot fetch LFS data.")
            return False
        ok = self.run("lfs_install", ["git", "lfs", "install", "--skip-repo"]) and \
            self.run("lfs_pull", ["git", "lfs", "pull", "--include", "SCANIA-Component-X/data/*"])
        return ok and (self.args.dry_run or self.scania_data_ready())

    def copy_results(self, sub: str):
        src = ML / "results" / sub
        if self.args.dry_run or not src.exists():
            return
        dst = self.out / "studies" / sub
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"copied {src} -> {dst}")

    def stage_studies(self) -> bool:
        banner("4/5  Optional studies")
        a = self.args
        if not (a.with_studies or a.with_scania or a.with_ablation):
            self.skip("studies", "none requested (--with-studies / --with-scania / --with-ablation)")
            return True
        if a.quick:
            self.skip("studies", "--quick (studies have no quick mode)")
            return True
        seeds = ["--seeds", *map(str, a.seeds)]
        ok = True
        if a.with_studies:
            ok &= self.run("condition_rul", [self.py, str(ML / "run_condition_rul.py"), *seeds])
            self.copy_results("family_comparison")
            ok &= self.run("n_cmapss_multi", [self.py, str(ML / "run_n_cmapss_multi.py"), *seeds])
            self.copy_results("family_comparison")
        if a.with_scania or a.with_ablation:
            if not self.ensure_scania():
                self.status.append({"stage": "scania_data", "status": "failed: could not fetch LFS data"})
                return False
        if a.with_scania:
            ok &= self.run("family_comparison", [self.py, str(ML / "run_family_comparison.py"), *seeds])
            ok &= self.run("scania_cost_tune", [self.py, str(ML / "run_scania_cost_tune.py"), *seeds])
            self.copy_results("family_comparison")
        if a.with_ablation:
            ok &= self.run("ablation", [self.py, str(ML / "run_ablation.py")])
            self.copy_results("ablation")
        return bool(ok)

    # ------------------------------------------------------------------ #
    def summary(self, t0: float) -> int:
        banner("5/5  Summary")
        failed = [s for s in self.status if str(s["status"]).startswith("failed")]
        for s in self.status:
            mins = f"{s['minutes']:.1f} min" if "minutes" in s else ""
            print(f"  {s['stage']:18s} {s['status']:28s} {mins}")
        lb = self.out / ("smoke" if self.args.quick else "benchmark") / "LEADERBOARD.md"
        if lb.exists():
            print(f"\nLeaderboard: {lb}")
        print(f"Logs:        {self.logs}")
        print(f"Total time:  {(time.time() - t0) / 60:.1f} min")
        if not self.args.dry_run:
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "pipeline_status.json").write_text(json.dumps({
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(self.args).items()},
                "stages": self.status,
            }, indent=2, default=str))
        print("\nPIPELINE " + ("FAILED" if failed else "COMPLETED"))
        return 1 if failed else 0

    def main(self) -> int:
        t0 = time.time()
        if not self.stage_env():
            return self.summary(t0)
        if not self.stage_smoke():
            print("Smoke test failed -> stopping before the long benchmark. See logs/smoke.log.")
            return self.summary(t0)
        bench_ok = self.stage_benchmark()
        if bench_ok or self.args.keep_going:
            self.stage_studies()
        return self.summary(t0)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ML / "results" / "pipeline"), help="output folder (e.g. a mounted Drive)")
    ap.add_argument("--quick", action="store_true", help="env + smoke test only (~2 min)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--datasets", nargs="+", help="benchmark datasets (default: all)")
    ap.add_argument("--models", nargs="+", help="benchmark models (default: all)")
    ap.add_argument("--epochs", type=int, help="override max epochs for deep models (default 40)")
    ap.add_argument("--skip-smoke", action="store_true", help="skip the smoke test")
    ap.add_argument("--with-studies", action="store_true", help="condition clustering + N-CMAPSS multi-file")
    ap.add_argument("--with-scania", action="store_true", help="SCANIA classification studies (pulls LFS data)")
    ap.add_argument("--with-ablation", action="store_true", help="LSTM ablation study (pulls LFS data)")
    ap.add_argument("--keep-going", action="store_true", help="run studies even if the benchmark failed")
    ap.add_argument("--install", action="store_true", help="pip install -r ml_pipeline/requirements.txt first")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(Pipeline(parse_args()).main())
