"""Command-line orchestration for the deep-momentum pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from changepoint_detection.progress import logged_stage

from .config import (
    DEFAULT_ARCHIVE,
    DEFAULT_BUILD_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MODEL_SPECS,
    Profile,
    load_profile,
)
from .detectors import prepare_detector_features
from .bootstrap import run_bootstrap
from .gate_by_class import run_gate_by_class
from .gating import (
    DEFAULT_GATED_MODELS,
    DEFAULT_P_GRID,
    DEFAULT_VARIANTS,
    run_gating,
)
from .features import prepare_base_features
from .reporting import (
    generate_reports,
    report_ensemble_size,
    validate_outputs,
    write_manifest,
)
from .tensors import prepare_tensor_caches
from .training import train_replicates, train_search


LOGGER = logging.getLogger(__name__)


def _paths(args: argparse.Namespace) -> tuple[Profile, Path, Path, Path]:
    """Resolve the profile and command paths from parsed arguments."""
    profile = load_profile(args.profile)
    archive = args.archive.resolve()
    build_dir = args.build_root.resolve() / profile.name
    output_dir = args.output_root.resolve() / profile.name
    build_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return profile, archive, build_dir, output_dir


def prepare_features(
    profile: Profile, archive: Path, build_dir: Path, workers: int
) -> None:
    """Construct base, detector, and tensor caches for a profile.

    Args:
        profile: Deep-momentum experiment profile.
        archive: Prepared continuous-futures ZIP archive.
        build_dir: Profile-specific cache directory.
        workers: Maximum number of contract feature jobs.
    """
    with logged_stage("Base price features", logger=LOGGER):
        prepare_base_features(archive, profile, build_dir, workers=workers)
    with logged_stage("Rolling BOCPD and CUSUM features", logger=LOGGER):
        prepare_detector_features(archive, profile, build_dir, workers=workers)
    with logged_stage("Shared model tensors", logger=LOGGER):
        prepare_tensor_caches(archive, profile, build_dir, workers=workers)


def run_command(args: argparse.Namespace) -> None:
    """Execute one deep-momentum command from an argparse namespace.

    Args:
        args: Validated command, profile, path, worker, ensemble, and gate options.
    """
    if args.command == "models":
        for spec in MODEL_SPECS:
            extras = spec.feature_columns[8:]
            print(
                f"{spec.name:24s} {len(spec.feature_columns):2d} inputs | "
                f"cost={spec.cost_bps:g}bp | {', '.join(extras) or 'base only'}"
            )
        return
    profile, archive, build_dir, output_dir = _paths(args)
    ensemble_size = report_ensemble_size(profile, args.ensemble_size)
    LOGGER.info(
        "Deep-momentum command=%s profile=%s feature_workers=%d train_workers=%d",
        args.command,
        profile.name,
        args.workers,
        args.train_workers,
    )
    LOGGER.info("Paths: archive=%s build=%s output=%s", archive, build_dir, output_dir)
    if args.command == "features":
        prepare_features(profile, archive, build_dir, args.workers)
    elif args.command == "train":
        with logged_stage("Random-search model training", logger=LOGGER):
            train_search(profile, build_dir, workers=args.train_workers)
    elif args.command == "replicate":
        with logged_stage("Selected-configuration seed replication", logger=LOGGER):
            train_replicates(profile, build_dir, workers=args.train_workers)
    elif args.command == "gate":
        with logged_stage("Post-hoc ensemble-uncertainty gating", logger=LOGGER):
            run_gating(
                profile,
                build_dir,
                output_dir,
                models=args.models,
                m_grid=args.m_grid,
                p_grid=args.p_grid,
                variants=args.variants,
                trial_members=args.gate_members or ensemble_size,
            )
    elif args.command == "gate-by-class":
        with logged_stage("Per-class uncertainty-gating analysis", logger=LOGGER):
            run_gate_by_class(
                profile,
                build_dir,
                output_dir,
                gate_members=args.gate_members or ensemble_size,
            )
    elif args.command == "bootstrap":
        with logged_stage("Paired gating bootstrap", logger=LOGGER):
            run_bootstrap(profile, output_dir, models=args.models)
    elif args.command == "report":
        with logged_stage("Deep-momentum report generation", logger=LOGGER):
            generate_reports(
                profile, build_dir, output_dir, ensemble=ensemble_size
            )
    elif args.command == "validate":
        with logged_stage("Deep-momentum output validation", logger=LOGGER):
            validation = validate_outputs(
                profile, build_dir, output_dir, ensemble=ensemble_size
            )
            write_manifest(
                profile,
                archive,
                build_dir,
                output_dir,
                validation,
                ensemble=ensemble_size,
            )
        print(json.dumps(validation, indent=2))
    elif args.command in {"smoke", "full"}:
        prepare_features(profile, archive, build_dir, args.workers)
        with logged_stage("Random-search model training", logger=LOGGER):
            train_search(profile, build_dir, workers=args.train_workers)
        with logged_stage("Selected-configuration seed replication", logger=LOGGER):
            train_replicates(profile, build_dir, workers=args.train_workers)
        with logged_stage("Deep-momentum report generation", logger=LOGGER):
            generate_reports(
                profile, build_dir, output_dir, ensemble=ensemble_size
            )
        with logged_stage("Post-hoc ensemble-uncertainty gating", logger=LOGGER):
            run_gating(
                profile,
                build_dir,
                output_dir,
                models=args.models,
                m_grid=args.m_grid,
                p_grid=args.p_grid,
                variants=args.variants,
                trial_members=args.gate_members or ensemble_size,
            )
        with logged_stage("Per-class uncertainty-gating analysis", logger=LOGGER):
            run_gate_by_class(
                profile,
                build_dir,
                output_dir,
                gate_members=args.gate_members or ensemble_size,
            )
        with logged_stage("Paired gating bootstrap", logger=LOGGER):
            run_bootstrap(profile, output_dir, models=args.models)
        with logged_stage("Deep-momentum output validation", logger=LOGGER):
            validation = validate_outputs(
                profile, build_dir, output_dir, ensemble=ensemble_size
            )
            write_manifest(
                profile,
                archive,
                build_dir,
                output_dir,
                validation,
                ensemble=ensemble_size,
            )
        print(json.dumps(validation, indent=2))
    else:
        raise AssertionError(args.command)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate deep-momentum command-line arguments.

    Args:
        argv: Argument vector excluding the program name, or ``None`` to use
            process arguments.

    Returns:
        Validated argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train deep momentum networks with changepoint features."
    )
    parser.add_argument(
        "command",
        choices=(
            "models",
            "features",
            "gate",
            "gate-by-class",
            "bootstrap",
            "train",
            "replicate",
            "report",
            "validate",
            "smoke",
            "full",
        ),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--gate-members",
        type=int,
        default=None,
        help="override the trial-kind gate ensemble size (default: top-5)",
    )
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        help="reporting ensemble size; does not alter feature or training caches",
    )
    def _model_list(value: str) -> tuple[str, ...]:
        """Parse a comma-separated model or variant list."""
        return tuple(item.strip() for item in value.split(",") if item.strip())

    def _float_list(value: str) -> tuple[float, ...]:
        """Parse a comma-separated numeric grid."""
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())

    parser.add_argument("--models", type=_model_list, default=DEFAULT_GATED_MODELS,
                        help="Models to gate (gate command only).")
    parser.add_argument("--m-grid", type=_float_list, default=None,
                        help="Override reallocation caps, e.g. 1,2,inf; default is "
                             "per-variant (gate only).")
    parser.add_argument("--variants", type=_model_list, default=DEFAULT_VARIANTS,
                        help="Gate variants: rank-hyst,rank-cwhyst (gate only).")
    parser.add_argument("--p-grid", type=_float_list, default=DEFAULT_P_GRID,
                        help="Target abstention rates, e.g. 0,0.1,0.25,0.5 (gate only).")
    default_workers = max(1, (os.cpu_count() or 1) - 2)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--train-workers", type=int, default=default_workers)
    args = parser.parse_args(argv)
    if args.command == "full":
        args.profile = "full"
    elif args.command == "smoke":
        args.profile = "smoke"
    if args.workers < 1 or args.train_workers < 1:
        parser.error("worker counts must be positive")
    if args.ensemble_size is not None and args.ensemble_size < 1:
        parser.error("ensemble size must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    """Configure logging and execute a deep-momentum command.

    Args:
        argv: Optional argument vector excluding the program name.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    run_command(parse_args(argv))
