"""
Unified command-line entrypoint for the AEMET radar pipeline.

Examples:
    py radar.py run --radar nacional
    py radar.py fetch --radar ba
    py radar.py visualize --radar se
    py radar.py enhance --radar ma
    py radar.py train --epochs 30 --patches 1000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config import AEMET_DEFAULT_RADAR, DATA_DIR, FETCH_DIR, ROOT_DIR, RUNS_DIR, ensure_core_dirs
from utils.logging_utils import configure_logging
from utils.metrics import read_metrics_json, write_metrics_json, write_summary_text


FETCH_SCRIPT = ROOT_DIR / "01_fetch_aemet.py"
VISUALIZE_SCRIPT = ROOT_DIR / "02_visualize.py"
ENHANCE_SCRIPT = ROOT_DIR / "03_enhance.py"
TRAIN_SCRIPT = ROOT_DIR / "04_superresolution.py"


def _radar_stem(radar: str) -> str:
    radar = radar.strip().lower()
    if radar == "nacional":
        return "AemetNacional"
    return f"Aemet{radar.upper()}"


def _latest_fetched_file(radar: str) -> Path | None:
    pattern = f"01_{_radar_stem(radar)}_*.gif"
    matches = list(FETCH_DIR.glob(pattern))

    legacy_pattern = f"aemet_{radar.lower()}_*.gif"
    matches.extend(DATA_DIR.glob(legacy_pattern))

    matches = sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    return None


def _run_script(script_path: Path, args: list[str], extra_env: dict[str, str] | None = None) -> None:
    cmd = [sys.executable, str(script_path), *args]
    print(f"\n==> Running: {' '.join(cmd)}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, cwd=ROOT_DIR, check=True, env=env)


def _resolve_target_file(radar: str, explicit_file: str | None) -> str | None:
    if explicit_file:
        return explicit_file

    latest = _latest_fetched_file(radar)
    if latest is None:
        print(f"No fetched GIF found for radar '{radar}'. Fetching a fresh file first.")
        _run_script(FETCH_SCRIPT, ["--radar", radar])
        latest = _latest_fetched_file(radar)

    if latest is None:
        raise FileNotFoundError(f"Could not find a fetched GIF for radar '{radar}'.")
    return latest.name


def command_fetch(args: argparse.Namespace) -> None:
    _run_script(FETCH_SCRIPT, ["--radar", args.radar])


def command_visualize(args: argparse.Namespace) -> None:
    target = _resolve_target_file(args.radar, args.file)
    script_args = ["-f", target] if target else []
    _run_script(VISUALIZE_SCRIPT, script_args)


def command_enhance(args: argparse.Namespace) -> None:
    target = _resolve_target_file(args.radar, args.file)
    script_args = ["-f", target] if target else []
    _run_script(ENHANCE_SCRIPT, script_args)


def command_train(args: argparse.Namespace) -> None:
    script_args = ["--epochs", str(args.epochs), "--patches", str(args.patches)]
    target = _resolve_target_file(args.radar, args.file) if args.verify else args.file
    if target:
        script_args.extend(["-f", target])
    _run_script(TRAIN_SCRIPT, script_args)


def command_run(args: argparse.Namespace) -> None:
    run_dir = _make_run_dir(args.radar)
    fetch_dir = run_dir / "01_fetch"
    visualize_dir = run_dir / "02_visualize"
    enhance_dir = run_dir / "03_enhance"
    for path in (fetch_dir, visualize_dir, enhance_dir):
        path.mkdir(parents=True, exist_ok=True)

    fetch_metrics = fetch_dir / "metrics.json"
    visualize_metrics = visualize_dir / "metrics.json"
    enhance_metrics = enhance_dir / "metrics.json"

    _run_script(
        FETCH_SCRIPT,
        ["--radar", args.radar],
        extra_env={
            "RADAR_STEP_OUT_DIR": str(fetch_dir),
            "RADAR_STEP_METRICS_JSON": str(fetch_metrics),
            "RADAR_STEP_SUMMARY_TXT": str(fetch_dir / "summary.txt"),
        },
    )
    target = _latest_run_fetch_file(fetch_dir)

    if args.visualize:
        _run_script(
            VISUALIZE_SCRIPT,
            ["-f", str(target)],
            extra_env={
                "RADAR_STEP_OUT_DIR": str(visualize_dir),
                "RADAR_STEP_METRICS_JSON": str(visualize_metrics),
                "RADAR_STEP_SUMMARY_TXT": str(visualize_dir / "summary.txt"),
                "RADAR_COMPARE_METRICS_JSON": str(fetch_metrics),
            },
        )
    _run_script(
        ENHANCE_SCRIPT,
        ["-f", str(target)],
        extra_env={
            "RADAR_STEP_OUT_DIR": str(enhance_dir),
            "RADAR_STEP_METRICS_JSON": str(enhance_metrics),
            "RADAR_STEP_SUMMARY_TXT": str(enhance_dir / "summary.txt"),
        },
    )
    _write_run_artifacts(run_dir, args.radar, target.name, args.visualize)
    print(f"\nRun artifacts saved to: {run_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified AEMET radar pipeline entrypoint",
        epilog=(
            "Examples:\n"
            "  py radar.py run --radar nacional\n"
            "  py radar.py enhance -file aemet_nacional_20260519_1039.gif\n"
            "  py radar.py visualize --radar ba\n"
            "  py radar.py train --epochs 30 --patches 1000\n\n"
            "Notes:\n"
            "  `-file` / `--file` is only valid with `visualize`, `enhance`, and `train`.\n"
            "  If no file is provided, the newest fetched GIF for the chosen radar is used."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch the latest AEMET radar GIF")
    fetch_parser.add_argument("--radar", default=AEMET_DEFAULT_RADAR, help="AEMET radar scope/code")
    fetch_parser.set_defaults(func=command_fetch)

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Visualize a fetched radar GIF",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    visualize_parser.add_argument("--radar", default=AEMET_DEFAULT_RADAR, help="AEMET radar scope/code")
    visualize_parser.add_argument(
        "-f", "-file", "--file",
        dest="file",
        default=None,
        help="Specific fetched GIF filename",
    )
    visualize_parser.set_defaults(func=command_visualize)

    enhance_parser = subparsers.add_parser(
        "enhance",
        help="Enhance a fetched radar GIF",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    enhance_parser.add_argument("--radar", default=AEMET_DEFAULT_RADAR, help="AEMET radar scope/code")
    enhance_parser.add_argument(
        "-f", "-file", "--file",
        dest="file",
        default=None,
        help="Specific fetched GIF filename",
    )
    enhance_parser.set_defaults(func=command_enhance)

    train_parser = subparsers.add_parser(
        "train",
        help="Train the super-resolution model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    train_parser.add_argument("--radar", default=AEMET_DEFAULT_RADAR, help="AEMET radar scope/code for verification")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--patches", type=int, default=500)
    train_parser.add_argument(
        "-f", "-file", "--file",
        dest="file",
        default=None,
        help="Specific fetched GIF filename for verification",
    )
    train_parser.add_argument("--no-verify", dest="verify", action="store_false", help="Skip real AEMET verification")
    train_parser.set_defaults(func=command_train, verify=True)

    run_parser = subparsers.add_parser("run", help="Fetch and enhance a radar in one command")
    run_parser.add_argument("--radar", default=AEMET_DEFAULT_RADAR, help="AEMET radar scope/code")
    run_parser.add_argument("--visualize", action="store_true", help="Also generate step-2 plots")
    run_parser.set_defaults(func=command_run)

    return parser


def main() -> None:
    ensure_core_dirs()
    configure_logging("radar")
    parser = build_parser()
    args = parser.parse_args()
    args.radar = args.radar.strip().lower()
    args.func(args)


def _make_run_dir(radar: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_{radar}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _latest_run_fetch_file(fetch_dir: Path) -> Path:
    matches = sorted(fetch_dir.glob("*.gif"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No fetched GIF found inside run folder: {fetch_dir}")
    return matches[0]


def _write_run_artifacts(run_dir: Path, radar: str, source_file: str, include_visualize: bool) -> None:
    payload = {
        "run_id": run_dir.name,
        "radar": radar,
        "source_file": source_file,
        "steps": {},
    }

    fetch_metrics_path = run_dir / "01_fetch" / "metrics.json"
    if fetch_metrics_path.exists():
        payload["steps"]["fetch"] = read_metrics_json(fetch_metrics_path)

    visualize_metrics_path = run_dir / "02_visualize" / "metrics.json"
    if include_visualize and visualize_metrics_path.exists():
        payload["steps"]["visualize"] = read_metrics_json(visualize_metrics_path)

    enhance_metrics_path = run_dir / "03_enhance" / "metrics.json"
    if enhance_metrics_path.exists():
        payload["steps"]["enhance"] = read_metrics_json(enhance_metrics_path)

    write_metrics_json(run_dir / "metrics.json", payload)

    summary_lines = [
        f"Run ID         : {run_dir.name}",
        f"Radar          : {radar}",
        f"Source file    : {source_file}",
    ]
    if "fetch" in payload["steps"]:
        fetch = payload["steps"]["fetch"]
        summary_lines.extend(
            [
                "",
                "--- Fetch ---",
                f"Signal pixels  : {fetch['signal_pixels']:,}",
                f"Value range    : {fetch['dbz_min']:.1f} ... {fetch['dbz_max']:.1f} dBZ",
            ]
        )
    if "visualize" in payload["steps"]:
        viz = payload["steps"]["visualize"]
        summary_lines.extend(
            [
                "",
                "--- Visualize ---",
                f"Signal pixels  : {viz['signal_pixels']:,}",
                f"Value range    : {viz['dbz_min']:.1f} ... {viz['dbz_max']:.1f} dBZ",
            ]
        )
    if "enhance" in payload["steps"]:
        enh = payload["steps"]["enhance"]
        summary_lines.extend(
            [
                "",
                "--- Enhance Raw ---",
                f"Signal pixels  : {enh['raw_metrics']['signal_pixels']:,}",
                f"Value range    : {enh['raw_metrics']['dbz_min']:.1f} ... {enh['raw_metrics']['dbz_max']:.1f} dBZ",
                "",
                "--- Enhance Output ---",
                f"Signal pixels  : {enh['enhanced_metrics']['signal_pixels']:,}",
                f"Value range    : {enh['enhanced_metrics']['dbz_min']:.1f} ... {enh['enhanced_metrics']['dbz_max']:.1f} dBZ",
                f"Diff mean      : {enh['difference_metrics']['mean_diff_dbz']:.2f} dBZ",
                f"Diff median    : {enh['difference_metrics']['median_diff_dbz']:.2f} dBZ",
            ]
        )

    write_summary_text(run_dir / "summary.txt", "\n".join(summary_lines))


if __name__ == "__main__":
    main()
