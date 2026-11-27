#!/usr/bin/env python3
"""
Run the whole pipeline with one command.

Stages (in order)
-----------------
  1. env        - Python / torch / GPU / dependency check (optionally pip install)
  2. data       - --with-ncmapss: download N-CMAPSS DS01/02/03/05/07 (~15 GB) from NASA,
                  streamed and CRC-checked; skipped if the files are already there
  3. smoke      - Continuous RUL Benchmark quick test: every model, 1 epoch (fails fast)
  4. benchmark  - full Continuous RUL Benchmark (CRB): all datasets x models x seeds
  5. studies    - optional extra experiments:
                    --with-studies   condition clustering (FD002/FD004) + N-CMAPSS multi-file
                    --with-scania    SCANIA classification (family comparison + cost tuning)
                    --with-ablation  LSTM one-factor ablations (uses SCANIA data too)
                  SCANIA data (Git LFS, ~1.7 GB) is pulled automatically when needed.
  6. summary    - status of every stage -> OUT/pipeline_status.json

  --all = --with-ncmapss --with-studies --with-scania --with-ablation (everything)

Everything is written under --out (default: ml_pipeline/results/pipeline):
  OUT/benchmark/   LEADERBOARD.md, runs.csv, leaderboard*.csv   (resumable)
  OUT/studies/     results of the optional studies (written there directly)
  OUT/logs/        one log file per stage
Re-running the same command resumes the benchmark (finished runs are skipped).

Examples
--------
  python run_pipeline.py                                   # env + smoke + full benchmark
  python run_pipeline.py --quick                           # ~2 min check of everything
  python run_pipeline.py --out /content/drive/MyDrive/CRB  # results on a mounted Google Drive
  python run_pipeline.py --models CATT+TCT CATT TCN --datasets FD001 FD004
  python run_pipeline.py --all --out /content/drive/MyDrive/CRB   # everything incl. N-CMAPSS
  python run_pipeline.py --with-ncmapss --ncmapss-dir /content/drive/MyDrive/CRB/N-CMAPSS  # keep data in Drive
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
ML = REPO / "ml_pipeline"
DEFAULT_NCMAPSS_DIR = REPO / "N-CMAPSS" / "data_set"
TCT_ABLATION_MODELS = ["CATT+TCT-noConsistency", "CATT+TCT-noLatePenalty", "CATT+TCT-noSmoothing",
                       "CATT+TCT-lambda0.3", "CATT+TCT-lambda3",
                       # smoothing-only baselines (plain model + causal smoothing, no TCT loss)
                       "CATT+Smooth", "TCN+Smooth", "LSTM+Smooth", "Transformer+Smooth"]
SCANIA_DATA = REPO / "SCANIA-Component-X" / "data"
SCANIA_LFS = ["train_operational_readouts.csv", "validation_operational_readouts.csv", "test_operational_readouts.csv"]

for _s in (sys.stdout, sys.stderr):  # never crash on console encoding
    try:
        _s.reconfigure(errors="replace")
    except AttributeError:
        pass


def _lfs_pointer(path: Path):
    """Return (sha256, size) if path is a Git LFS pointer file, else None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(200)
    except OSError:
        return None
    if not head.startswith(b"version https://git-lfs"):
        return None
    txt = head.decode("utf-8", "replace")
    oid = next((l.split("sha256:", 1)[1].strip() for l in txt.splitlines() if l.startswith("oid sha256:")), None)
    size = next((int(l.split()[1]) for l in txt.splitlines() if l.startswith("size ")), None)
    return (oid, size) if oid and size else None


def _github_slug_and_ref(repo: Path):
    """owner/name and commit of the clone (for media.githubusercontent.com); safe defaults."""
    slug, ref = "Nafis878/Continuous-", "main"
    try:
        url = subprocess.run(["git", "-C", str(repo), "remote", "get-url", "origin"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        if "github.com" in url:
            slug = url.split("github.com", 1)[1].lstrip(":/").removesuffix(".git").split("@")[-1]
            slug = "/".join(slug.split("/")[-2:])
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
        if len(head) == 40:
            ref = head
    except Exception:
        pass
    return slug, ref


def download_lfs_file(repo: Path, rel_path: str, log=print) -> bool:
    """
    Replace a Git LFS pointer with the real file by downloading it from GitHub's media
    server (works without git-lfs installed). Size and SHA-256 are verified.
    """
    import hashlib
    import urllib.request

    target = repo / rel_path
    ptr = _lfs_pointer(target)
    if ptr is None:
        return target.exists()
    oid, size = ptr
    slug, ref = _github_slug_and_ref(repo)
    url = f"https://media.githubusercontent.com/media/{slug}/{ref}/{rel_path}"
    tmp = target.with_name(target.name + ".part")
    for attempt in range(5):
        try:
            log(f"[lfs] downloading {rel_path} ({size / 1e6:.0f} MB) from {url}")
            h, n = hashlib.sha256(), 0
            req = urllib.request.Request(url, headers={"User-Agent": "crb-pipeline"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(8 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
            if n != size or h.hexdigest() != oid:
                raise IOError(f"verification failed (size {n}/{size}, sha256 {h.hexdigest()[:12]}/{oid[:12]})")
            os.replace(tmp, target)
            log(f"[lfs] OK {rel_path} (size + sha256 verified)")
            return True
        except Exception as exc:
            log(f"[lfs] attempt {attempt + 1} failed: {exc}")
            time.sleep(min(60, 5 * 2 ** attempt))
    tmp.unlink(missing_ok=True)
    return False


# study name -> output file that only exists when the study finished (under OUT/studies)
STUDY_DONE = {
    "condition_rul": "family_comparison/condition_rul_summary.json",
    "n_cmapss_multi": "family_comparison/n_cmapss_multi_summary.json",
    "family_comparison": "family_comparison/summary.json",
    "scania_cost_tune": "family_comparison/scania_cost_tune.json",
    "ablation": "ablation/ablation_summary.json",
}


def banner(text: str):
    print("\n" + "=" * 78 + f"\n  {text}\n" + "=" * 78, flush=True)


DRIVE_MOUNT = "/content/drive"
LOCAL_WORK = "/content/crb_work"


def _under_drive(p) -> bool:
    return str(p).replace("\\", "/").startswith(DRIVE_MOUNT + "/")


class Mirror:
    """
    Copies the local work folder to the final output folder (e.g. Google Drive).

    Colab's Drive mount can drop mid-run ("[Errno 107] Transport endpoint is not
    connected"). The pipeline therefore always writes to local disk and this
    mirror copies changed files to Drive periodically and after every stage.
    Copy errors never stop the pipeline; they are reported and retried later.
    Files are copied to "<name>.synctmp" and renamed, so Drive never holds a
    half-written file.
    """

    def __init__(self, work: Path, final: Path, every: int):
        self.work, self.final, self.every = work, final, every
        self.enabled = work.resolve() != final.resolve()
        self.ok = True
        self.last_error: str | None = None
        self.last_sync: float | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _copy_tree(src: Path, dst: Path, only_newer: bool = True) -> int:
        n = 0
        for f in src.rglob("*"):
            if not f.is_file() or f.name.endswith(".synctmp"):
                continue
            t = dst / f.relative_to(src)
            st = f.stat()
            if only_newer and t.exists():
                tt = t.stat()
                if tt.st_size == st.st_size and tt.st_mtime >= st.st_mtime - 1:
                    continue
            t.parent.mkdir(parents=True, exist_ok=True)
            tmp = t.with_name(t.name + ".synctmp")
            shutil.copy2(f, tmp)
            os.replace(tmp, t)
            n += 1
        return n

    def seed_from_final(self):
        """Resume support: bring earlier results (e.g. from Drive) into the local work folder."""
        if not self.enabled or not self.final.exists():
            return
        try:
            n = self._copy_tree(self.final, self.work)
            if n:
                print(f"[sync] restored {n} file(s) from {self.final} -> {self.work} (resuming)", flush=True)
        except OSError as exc:
            print(f"[sync] WARNING: could not read earlier results from {self.final}: {exc}", flush=True)

    def sync(self, reason: str = "") -> bool:
        if not self.enabled:
            return True
        with self._lock:
            try:
                n = self._copy_tree(self.work, self.final)
                if not self.ok:
                    print(f"[sync] Drive reachable again - results synced to {self.final}", flush=True)
                self.ok, self.last_error, self.last_sync = True, None, time.time()
                if n and reason:
                    print(f"[sync] {n} file(s) -> {self.final} ({reason})", flush=True)
                return True
            except OSError as exc:
                if self.ok:
                    print(f"[sync] WARNING: cannot write to {self.final}: {exc}\n"
                          f"[sync] The pipeline keeps running; results are safe in {self.work} and "
                          f"will be copied when the folder is reachable again.", flush=True)
                self.ok, self.last_error = False, str(exc)
                return False

    def start(self):
        if not self.enabled or self.every <= 0:
            return

        def loop():
            while not self._stop.wait(self.every):
                self.sync()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.final = Path(args.out).resolve()
        if args.work_dir:
            self.out = Path(args.work_dir).resolve()
        elif _under_drive(args.out):
            self.out = Path(LOCAL_WORK)  # write locally, mirror to Drive
        else:
            self.out = self.final
        self.mirror = Mirror(self.out, self.final, args.sync_every)
        self.logs = self.out / "logs"
        self.status: list[dict] = []
        self.py = sys.executable
        self.ncmapss_dir = Path(args.ncmapss_dir).resolve() if args.ncmapss_dir else DEFAULT_NCMAPSS_DIR
        # every subprocess writes study results under OUT/studies and finds N-CMAPSS here
        self.child_env = {"CRB_RESULTS_DIR": str(self.out / "studies"), "NCMAPSS_DIR": str(self.ncmapss_dir),
                          "CRB_RF_BACKEND": args.rf_backend}

    # ------------------------------------------------------------------ #
    def run(self, name: str, cmd: list[str], env: dict | None = None, tag: str | None = None) -> bool:
        """Run a command, stream its output live and tee it to OUT/logs/<name>.log.
        tag: prefix for console lines when several commands run in parallel."""
        pre = f"[{tag}:{name}] " if tag else ""
        print(f"{pre}$ {' '.join(cmd)}", flush=True)
        if self.args.dry_run:
            self.status.append({"stage": name, "status": "dry-run", "cmd": cmd})
            return True
        self.logs.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        full_env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                    **self.child_env, **(env or {})}
        with open(self.logs / f"{name}.log", "w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen(cmd, cwd=REPO, env=full_env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                print(pre + line, end="", flush=True)
                log.write(line)
            rc = proc.wait()
        ok = rc == 0
        self.status.append({"stage": name, "status": "ok" if ok else f"failed (exit {rc})",
                            "minutes": round((time.time() - t0) / 60, 2), "cmd": cmd})
        print(f"--> {name}: {'OK' if ok else f'FAILED (exit {rc})'} in {(time.time() - t0) / 60:.1f} min", flush=True)
        self.mirror.sync(f"after {name}")
        return ok

    def skip(self, name: str, why: str):
        print(f"--> {name}: skipped ({why})", flush=True)
        self.status.append({"stage": name, "status": f"skipped: {why}"})

    # ------------------------------------------------------------------ #
    def stage_env(self) -> bool:
        banner("1/8  Environment")
        if self.args.install and not self.args.dry_run:
            if not self.run("pip_install", [self.py, "-m", "pip", "install", "-q", "-r", str(ML / "requirements.txt")]):
                return False
        print(f"python  {platform.python_version()} ({sys.executable})")
        print(f"repo    {REPO}")
        print(f"output  {self.final}")
        if self.mirror.enabled:
            print(f"work    {self.out}  (local; copied to output every {self.args.sync_every}s and after each stage)")
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
        n_h5 = len(list(self.ncmapss_dir.glob("N-CMAPSS_*.h5"))) if self.ncmapss_dir.exists() else 0
        print(f"N-CMAPSS {self.ncmapss_dir}  ({n_h5} .h5 file(s) present"
              f"{'; will download' if self.args.with_ncmapss else '; N-CMAPSS skipped unless present'})")
        self.status.append({"stage": "env", "status": "ok", "gpu": torch.cuda.is_available()})
        return True

    def stage_data(self) -> bool:
        banner("2/8  Data (N-CMAPSS)")
        if not self.args.with_ncmapss:
            self.skip("ncmapss_download", "not requested (--with-ncmapss / --all)")
            return True
        return self.run("ncmapss_download", [self.py, str(ML / "datasets" / "ncmapss_download.py"),
                                             "--dest", str(self.ncmapss_dir)])

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
        banner("3/8  Smoke test (every model, 1 epoch, small subsets)")
        if self.args.skip_smoke and not self.args.quick:
            self.skip("smoke", "--skip-smoke")
            return True
        # benchmark RandomForest is always sklearn (as in the published runs); GPU RF is for studies only
        return self.run("smoke", self.bench_cmd(self.out / "smoke", quick=True), env={"CRB_RF_BACKEND": "sklearn"})

    def stage_benchmark(self) -> bool:
        banner("4/8  Continuous RUL Benchmark (CRB)")
        if self.args.quick:
            self.skip("benchmark", "--quick (smoke test only)")
            return True
        return self.run("benchmark", self.bench_cmd(self.out / "benchmark", quick=False),
                        env={"CRB_RF_BACKEND": "sklearn"})

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
        if self.args.dry_run:
            print("(dry-run) would fetch SCANIA LFS data")
            return True
        has_lfs = shutil.which("git") is not None and subprocess.run(
            ["git", "lfs", "version"], capture_output=True).returncode == 0
        if has_lfs:
            self.run("lfs_pull", ["git", "lfs", "pull", "--include", "SCANIA-Component-X/data/*"])
            if self.scania_data_ready():
                return True
            print("git lfs pull did not fetch the files -> direct download fallback")
        else:
            print("git-lfs is not installed -> downloading the files directly from GitHub")
        ok = all(download_lfs_file(REPO, f"SCANIA-Component-X/data/{f}") for f in SCANIA_LFS)
        self.status.append({"stage": "scania_data", "status": "ok" if ok else "failed: download"})
        return ok and self.scania_data_ready()

    def stage_studies(self) -> bool:
        banner("6/8  Optional studies")
        a = self.args
        if not (a.with_studies or a.with_scania or a.with_ablation):
            self.skip("studies", "none requested (--with-studies / --with-scania / --with-ablation)")
            return True
        if a.quick:
            self.skip("studies", "--quick (studies have no quick mode)")
            return True
        seeds = ["--seeds", *map(str, a.study_seeds)]
        print(f"RandomForest backend: {a.rf_backend}  |  parallel study lanes: {a.parallel_studies}  |  "
              f"study seeds: {a.study_seeds}", flush=True)

        checked: dict[str, bool] = {}

        def done(name):
            if name in checked:  # report each skipped study once
                return checked[name]
            res = False
            if not a.rerun_studies and (self.out / "studies" / STUDY_DONE[name]).exists():
                self.skip(name, f"already complete ({STUDY_DONE[name]} exists; --rerun-studies to redo)")
                res = True
            checked[name] = res
            return res

        need_scania = (a.with_scania and not (done("family_comparison") & done("scania_cost_tune"))) or \
            (a.with_ablation and not done("ablation"))
        if need_scania and not self.ensure_scania():
            self.status.append({"stage": "scania_data", "status": "failed: could not fetch SCANIA data"})
            print("SCANIA data unavailable -> running only the studies that do not need it.")

        # Two independent lanes (different result files, moderate RAM); the ablation (loads
        # SCANIA-TTE + all RUL sets, heaviest on RAM) always runs alone afterwards.
        scania_ok = self.scania_data_ready()
        lane_a = []
        if a.with_studies:
            lane_a += [(n, [self.py, str(ML / f"run_{n}.py"), *seeds])
                       for n in ("condition_rul", "n_cmapss_multi") if not done(n)]
        lane_b = []
        if a.with_scania:
            for n in ("family_comparison", "scania_cost_tune"):
                if done(n):
                    continue
                if not scania_ok:
                    self.status.append({"stage": n, "status": "failed: SCANIA data missing"})
                    continue
                lane_b.append((n, [self.py, str(ML / f"run_{n}.py"), *seeds]))

        def run_lane(lane, tag):
            res = True
            for name, cmd in lane:
                res &= self.run(name, cmd, tag=tag)
            return res

        ok = True
        if a.parallel_studies > 1 and lane_a and lane_b and not a.dry_run:
            print("Running lanes in parallel: A=[condition_rul, n_cmapss_multi]  "
                  "B=[family_comparison, scania_cost_tune]", flush=True)
            with ThreadPoolExecutor(max_workers=2) as ex:
                fa = ex.submit(run_lane, lane_a, "A")
                fb = ex.submit(run_lane, lane_b, "B")
                ok = fa.result() and fb.result()
        else:
            ok = run_lane(lane_a, None) and ok
            ok = run_lane(lane_b, None) and ok
        if a.with_ablation and not done("ablation"):
            if scania_ok:
                ok = self.run("ablation", [self.py, str(ML / "run_ablation.py")]) and ok
            else:
                self.status.append({"stage": "ablation", "status": "failed: SCANIA data missing"})
                ok = False
        return bool(ok)

    # ------------------------------------------------------------------ #
    def stage_tct_ablation(self) -> bool:
        banner("5/8  TCT component ablation")
        if not self.args.with_tct_ablation:
            self.skip("tct_ablation", "not requested (--with-tct-ablation / --all)")
            return True
        if self.args.quick:
            self.skip("tct_ablation", "--quick")
            return True
        cmd = [self.py, str(ML / "run_benchmark.py"), "--out", str(self.out / "tct_ablation"),
               "--seeds", *map(str, self.args.seeds), "--models", *TCT_ABLATION_MODELS]
        if self.args.datasets:
            cmd += ["--datasets", *self.args.datasets]
        if self.args.epochs:
            cmd += ["--epochs", str(self.args.epochs)]
        return self.run("tct_ablation", cmd)

    def stage_analysis(self) -> bool:
        banner("7/8  Statistical analysis")
        runs = self.out / ("smoke" if self.args.quick else "benchmark") / "runs.csv"
        if self.args.no_analysis or self.args.quick:
            self.skip("analysis", "--no-analysis" if self.args.no_analysis else "--quick")
            return True
        if not runs.exists() and not self.args.dry_run:
            self.skip("analysis", f"{runs} not found")
            return True
        cmd = [self.py, str(ML / "analyze_results.py"), "--runs", str(runs), "--out", str(self.out / "analysis")]
        abl = self.out / "tct_ablation" / "runs.csv"
        if abl.exists() or (self.args.dry_run and self.args.with_tct_ablation):
            cmd += ["--ablation-runs", str(abl)]
        return self.run("analysis", cmd)

    # ------------------------------------------------------------------ #
    def summary(self, t0: float) -> int:
        banner("8/8  Summary")
        failed = [s for s in self.status if str(s["status"]).startswith("failed")]
        for s in self.status:
            mins = f"{s['minutes']:.1f} min" if "minutes" in s else ""
            print(f"  {s['stage']:18s} {s['status']:28s} {mins}")
        sub = ("smoke" if self.args.quick else "benchmark") + "/LEADERBOARD.md"
        if (self.out / sub).exists():
            print(f"\nLeaderboard: {self.final / sub}")
        print(f"Logs:        {self.final / 'logs'}")
        print(f"Total time:  {(time.time() - t0) / 60:.1f} min")
        if not self.args.dry_run:
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "pipeline_status.json").write_text(json.dumps({
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(self.args).items()},
                "stages": self.status,
            }, indent=2, default=str))
            self.mirror.stop()
            if self.mirror.enabled and not self.mirror.sync("final"):
                print(f"\nWARNING: could not copy results to {self.final} ({self.mirror.last_error}).\n"
                      f"They are safe in {self.out} (until the runtime is reset). Re-mount Google Drive:\n"
                      "    from google.colab import drive; drive.flush_and_unmount(); "
                      "drive.mount('/content/drive', force_remount=True)\n"
                      "then copy them with:\n"
                      f"    python run_pipeline.py --sync-only --out \"{self.final}\"")
                failed = failed or [{"stage": "sync"}]
            elif self.mirror.enabled:
                print(f"All results copied to {self.final}")
        print("\nPIPELINE " + ("FAILED" if failed else "COMPLETED"))
        return 1 if failed else 0

    def seed_from_results(self, src_arg: str):
        """Start from earlier results (e.g. committed in results/<run>/): finished runs are skipped."""
        src = Path(src_arg)
        if not src.is_absolute() and not src.exists():
            src = REPO / src_arg  # allow paths relative to the repo root
        if not src.is_dir():
            print(f"[resume] WARNING: --resume-from {src_arg} not found -> starting without it", flush=True)
            return
        n = Mirror._copy_tree(src, self.out)
        print(f"[resume] seeded {n} file(s) from {src} -> {self.out}", flush=True)

    def main(self) -> int:
        t0 = time.time()
        if not self.args.dry_run:
            self.out.mkdir(parents=True, exist_ok=True)
            if self.args.resume_from:
                self.seed_from_results(self.args.resume_from)
            self.mirror.seed_from_final()
            if self.args.sync_only:
                ok = self.mirror.sync("sync-only") if self.mirror.enabled else True
                print(f"Synced {self.out} -> {self.final}" if ok else f"Sync FAILED: {self.mirror.last_error}")
                return 0 if ok else 1
            self.mirror.start()
        if not self.stage_env():
            return self.summary(t0)
        if not self.stage_data():
            print("N-CMAPSS download failed -> stopping. See logs/ncmapss_download.log (re-run to resume).")
            return self.summary(t0)
        if not self.stage_smoke():
            print("Smoke test failed -> stopping before the long benchmark. See logs/smoke.log.")
            return self.summary(t0)
        bench_ok = self.stage_benchmark()
        if bench_ok or self.args.keep_going:
            self.stage_tct_ablation()
            self.stage_studies()
        self.stage_analysis()
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
    ap.add_argument("--all", action="store_true",
                    help="everything: --with-ncmapss --with-studies --with-scania --with-ablation")
    ap.add_argument("--with-ncmapss", action="store_true", help="download N-CMAPSS (~15 GB) and include it")
    ap.add_argument("--ncmapss-dir", help=f"where N-CMAPSS .h5 files live / are downloaded (default {DEFAULT_NCMAPSS_DIR})")
    ap.add_argument("--with-tct-ablation", action="store_true",
                    help="TCT component ablation (CATT+TCT variants) + smoothing-only baselines -> OUT/tct_ablation")
    ap.add_argument("--study-seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="seeds for the optional studies (default 42 43 44)")
    ap.add_argument("--rerun-studies", action="store_true", help="re-run studies that are already complete")
    ap.add_argument("--no-analysis", action="store_true", help="skip the statistical analysis stage")
    ap.add_argument("--with-studies", action="store_true", help="condition clustering + N-CMAPSS multi-file")
    ap.add_argument("--with-scania", action="store_true", help="SCANIA classification studies (pulls LFS data)")
    ap.add_argument("--with-ablation", action="store_true", help="LSTM ablation study (pulls LFS data)")
    ap.add_argument("--keep-going", action="store_true", help="run studies even if the benchmark failed")
    ap.add_argument("--install", action="store_true", help="pip install -r ml_pipeline/requirements.txt first")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--work-dir", help=f"local working folder (default: {LOCAL_WORK} when --out is on "
                                       "Google Drive, otherwise --out itself)")
    ap.add_argument("--sync-every", type=int, default=120, help="seconds between copies work -> --out (default 120)")
    ap.add_argument("--sync-only", action="store_true", help="only copy the work folder to --out, then exit")
    ap.add_argument("--rf-backend", choices=["sklearn", "gpu"], default="sklearn",
                    help="RandomForest in the optional STUDIES: sklearn on CPU (default) or gpu = XGBoost "
                         "random forest on CUDA (much faster on Colab T4). The benchmark always uses sklearn.")
    ap.add_argument("--parallel-studies", type=int, choices=[1, 2], default=1,
                    help="run the two independent study lanes at the same time (2) or one after another (1)")
    ap.add_argument("--resume-from", help="earlier results folder to start from, e.g. results/crb_2026-11-27_T4_full "
                                          "(finished benchmark runs are skipped)")
    a = ap.parse_args(argv)
    if a.all:
        a.with_ncmapss = a.with_studies = a.with_scania = a.with_ablation = a.with_tct_ablation = True
    return a


def check_drive_paths(args) -> str | None:
    """
    Refuse to use /content/drive/... unless Google Drive is mounted there AND reachable.
    Not mounted -> results would go to a plain local folder and a later drive.mount() fails
    with "Mountpoint must not already contain files". Mounted but broken -> "[Errno 107]
    Transport endpoint is not connected".
    """
    paths = [p for p in (args.out, args.ncmapss_dir) if p]
    if not any(_under_drive(p) for p in paths):
        return None
    if os.path.ismount(DRIVE_MOUNT):
        try:
            os.listdir(DRIVE_MOUNT)
            return None
        except OSError as exc:
            return (
                f"Google Drive is mounted at {DRIVE_MOUNT} but not reachable ({exc}).\n"
                "The Drive connection dropped. Re-mount it in a Colab cell, then re-run the pipeline\n"
                "(finished work is resumed; downloaded N-CMAPSS files are reused):\n\n"
                "    from google.colab import drive\n"
                "    try:\n"
                "        drive.flush_and_unmount()\n"
                "    except Exception:\n"
                "        pass\n"
                f"    drive.mount('{DRIVE_MOUNT}', force_remount=True)\n"
            )
    return (
        f"--out / --ncmapss-dir point into {DRIVE_MOUNT}, but Google Drive is NOT mounted there.\n"
        "Run this in a Colab cell first, then re-run the pipeline:\n\n"
        "    import os, shutil\n"
        "    from google.colab import drive\n"
        f"    if not os.path.ismount('{DRIVE_MOUNT}'):\n"
        f"        shutil.rmtree('{DRIVE_MOUNT}', ignore_errors=True)  # local leftover, not your Drive\n"
        f"        drive.mount('{DRIVE_MOUNT}')\n"
    )


if __name__ == "__main__":
    _args = parse_args()
    _err = None if _args.dry_run else check_drive_paths(_args)
    if _err:
        print("ERROR: " + _err, file=sys.stderr)
        sys.exit(2)
    sys.exit(Pipeline(_args).main())
