"""YAML-driven experiment runner with GPU auto-selection and manifest tracking.

    python -m iosp.run_experiment configs/multistart_robustness.yaml
    python -m iosp.run_experiment configs/multistart_robustness.yaml --run joint_seed0
    python -m iosp.run_experiment configs/multistart_robustness.yaml --figures-only
    python -m iosp.run_experiment --all
    python -m iosp.run_experiment --all --dry-run
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import yaml

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "experiments", "configs")


def _config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_free_gpus(max_mem_mib=100):
    """Return list of GPU indices with < max_mem_mib memory used."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:
        return []
    free = []
    for line in out.strip().split("\n"):
        parts = line.split(",")
        if len(parts) == 2:
            idx, mem = int(parts[0].strip()), float(parts[1].strip())
            if mem < max_mem_mib:
                free.append(idx)
    return free


def pick_gpu(env_override=None):
    """Pick a free GPU. Respects CUDA_VISIBLE_DEVICES if set to a specific index."""
    if env_override and env_override != "auto":
        return str(env_override)
    free = get_free_gpus()
    if not free:
        raise RuntimeError("No free GPUs (all have >100 MiB in use). Aborting.")
    return str(free[0])


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_cmd(module, defaults, run_args):
    """Build a command list from module + merged defaults/run args."""
    cmd = [sys.executable, "-m", module]
    merged = dict(defaults or {})
    merged.update(run_args or {})
    for k, v in merged.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    return cmd


def build_env(cfg_env):
    """Merge config env vars into the current environment."""
    env = os.environ.copy()
    for k, v in (cfg_env or {}).items():
        if k == "CUDA_VISIBLE_DEVICES" and v == "auto":
            continue
        env[k] = str(v)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def write_manifest(out_path, cfg, run_entry, gpu_id, wall_secs, returncode):
    manifest = {
        "config_hash": _config_hash(cfg),
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu_id": gpu_id,
        "wall_clock_seconds": round(wall_secs, 1),
        "returncode": returncode,
        "run": run_entry.get("name", "unknown"),
        "out": run_entry.get("out", ""),
    }
    manifest_path = out_path.rsplit(".", 1)[0] + ".manifest.json" if out_path else None
    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  manifest: {manifest_path}")


def run_single(cfg, run_entry, gpu_id, dry_run=False):
    """Run one experiment entry."""
    module = cfg["module"]
    cmd = build_cmd(module, cfg.get("defaults"), run_entry.get("args"))
    if "out" in run_entry:
        cmd.extend(["--out", run_entry["out"]])
        os.makedirs(os.path.dirname(run_entry["out"]) or ".", exist_ok=True)

    env = build_env(cfg.get("env"))
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    name = run_entry.get("name", "?")
    print(f"\n{'='*60}")
    print(f"[run] {name}  GPU={gpu_id}  module={module}")
    print(f"  cmd: {' '.join(cmd)}")
    if dry_run:
        print("  (dry run — skipped)")
        return 0

    t0 = time.perf_counter()
    result = subprocess.run(cmd, env=env)
    wall = time.perf_counter() - t0
    print(f"  wall: {wall:.0f}s  exit: {result.returncode}")

    if run_entry.get("out"):
        write_manifest(run_entry["out"], cfg, run_entry, gpu_id, wall, result.returncode)
    return result.returncode


def run_renderers(cfg, kind, dry_run=False):
    """Run figure or table renderers defined in the config."""
    entries = cfg.get(kind, [])
    for entry in entries:
        renderer = entry["renderer"]
        cmd = [sys.executable, "-c",
               f"from {renderer} import render; "
               f"render({entry.get('input', entry.get('input_dir'))!r}, {entry['out']!r})"]
        print(f"\n[{kind}] {entry.get('name', '?')}: {renderer} -> {entry['out']}")
        if dry_run:
            print("  (dry run — skipped)")
            continue
        os.makedirs(os.path.dirname(entry["out"]) or ".", exist_ok=True)
        subprocess.run(cmd)


def run_config(config_path, run_name=None, figures_only=False, dry_run=False,
               gpu_override=None):
    """Run all (or one) entries from a YAML config."""
    cfg = load_config(config_path)
    print(f"\n{'#'*60}")
    print(f"# {cfg.get('experiment', config_path)}: {cfg.get('description', '')}")
    print(f"{'#'*60}")

    if figures_only:
        run_renderers(cfg, "figures", dry_run)
        run_renderers(cfg, "tables", dry_run)
        return

    gpu_id = gpu_override or pick_gpu(cfg.get("env", {}).get("CUDA_VISIBLE_DEVICES"))

    runs = cfg.get("runs", [])
    if run_name:
        runs = [r for r in runs if r.get("name") == run_name]
        if not runs:
            print(f"ERROR: no run named '{run_name}' in {config_path}")
            sys.exit(1)

    failures = 0
    for entry in runs:
        rc = run_single(cfg, entry, gpu_id, dry_run)
        if rc != 0:
            failures += 1

    run_renderers(cfg, "figures", dry_run)
    run_renderers(cfg, "tables", dry_run)

    if failures:
        print(f"\nWARNING: {failures}/{len(runs)} runs failed")
    return failures


def main():
    ap = argparse.ArgumentParser(description="IOSP experiment runner")
    ap.add_argument("config", nargs="?", help="Path to YAML config")
    ap.add_argument("--run", dest="run_name", help="Run only this named entry")
    ap.add_argument("--figures-only", action="store_true",
                    help="Only run figure/table renderers, skip experiments")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing")
    ap.add_argument("--all", action="store_true",
                    help="Run all configs in experiments/configs/")
    ap.add_argument("--gpu", type=str, default=None,
                    help="Override GPU index (e.g. '0' or '2')")
    args = ap.parse_args()

    if args.all:
        configs = sorted(glob.glob(os.path.join(CONFIGS_DIR, "*.yaml")))
        if not configs:
            print(f"No configs found in {CONFIGS_DIR}")
            sys.exit(1)
        total_failures = 0
        for cfg_path in configs:
            total_failures += run_config(cfg_path, dry_run=args.dry_run,
                                         gpu_override=args.gpu) or 0
        if total_failures:
            print(f"\n{total_failures} total failures across all configs")
            sys.exit(1)
    elif args.config:
        config_path = args.config
        if not os.path.isabs(config_path) and not os.path.exists(config_path):
            config_path = os.path.join(CONFIGS_DIR, config_path)
        failures = run_config(config_path, run_name=args.run_name,
                              figures_only=args.figures_only,
                              dry_run=args.dry_run, gpu_override=args.gpu)
        if failures:
            sys.exit(1)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
