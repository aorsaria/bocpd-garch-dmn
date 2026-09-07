"""Command-line orchestration for changepoint experiments."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .config import (
    DEFAULT_ARCHIVE,
    DEFAULT_BUILD_ROOT,
    DEFAULT_OUTPUT_ROOT,
    ExperimentConfig,
    PROJECT_ROOT,
    load_profile,
)
from .data import (
    TrainingInputs,
    TrainingInputCacheMismatch,
    build_training_inputs,
    load_training_inputs,
    save_training_inputs,
)
from .progress import logged_stage
from .real_data import run_event_study, select_nu
from .reporting import validate_outputs, write_manifest
from .simulation import run_simulation
from .storage import clean_changepoint_artifacts

LOGGER = logging.getLogger(__name__)


def _paths(args: argparse.Namespace) -> tuple[ExperimentConfig, Path, Path, Path]:
    """Resolve the selected profile and all command paths from parsed arguments."""
    config = load_profile(args.profile)
    archive = args.archive.resolve()
    output_dir = args.output_root.resolve() / config.name
    build_dir = args.build_root.resolve() / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    return config, archive, output_dir, build_dir


def _training_inputs(
    config: ExperimentConfig, archive: Path, output_dir: Path, *, force: bool = False
) -> TrainingInputs:
    """Load cached market inputs or fit them when unavailable or explicitly forced."""
    if not force:
        try:
            return load_training_inputs(archive, output_dir)
        except (FileNotFoundError, TrainingInputCacheMismatch) as error:
            LOGGER.info("Cached training inputs cannot be used: %s", error)
    else:
        LOGGER.info("Forced refit requested; cached training inputs will not be used")
    LOGGER.info(
        "Fitting GARCH priors with %d initial starts per contract",
        config.garch_fit_starts,
    )
    inputs = build_training_inputs(
        archive, starts=config.garch_fit_starts, seed=config.seed + 100_000
    )
    save_training_inputs(inputs, output_dir)
    return inputs


def _selected_nu(output_dir: Path) -> float:
    """Read the selected Student-t degrees of freedom from a completed stage."""
    path = output_dir / "nu_selected.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run select-nu first")
    return float(json.loads(path.read_text())["selected_nu"])


def _finalize(
    config: ExperimentConfig,
    archive: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Validate BOCPD outputs, write their manifest, and print the summary."""
    with logged_stage("Output structural validation", logger=LOGGER):
        validation = validate_outputs(config, output_dir)
    with logged_stage("Output manifest", logger=LOGGER):
        write_manifest(config, archive, output_dir, validation)
    result = dict(validation)
    print(json.dumps(result, indent=2))
    return result


def run_command(args: argparse.Namespace) -> None:
    """Execute one BOCPD command from an argparse namespace.

    Args:
        args: Validated command, profile, path, worker, and cache options.
    """
    if args.command == "clean":
        build, output = clean_changepoint_artifacts(PROJECT_ROOT)
        LOGGER.info("Removed changepoint build artifacts from %s", build)
        LOGGER.info("Removed changepoint output artifacts from %s", output)
        return

    config, archive, output_dir, build_dir = _paths(args)
    command = args.command
    LOGGER.info(
        "Configuration: command=%s | profile=%s | workers=%d | particles=%d | "
        "max_run_lengths=%d",
        command,
        config.name,
        args.workers,
        config.particles,
        config.max_run_lengths,
    )
    LOGGER.info(
        "Paths: archive=%s | output=%s | build=%s", archive, output_dir, build_dir
    )
    if not archive.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive}")

    if command == "simulation":
        with logged_stage("Simulation and continuous-monitoring study", logger=LOGGER):
            run_simulation(config, output_dir, build_dir, workers=args.workers)
        return
    if command in {"validate", "finalize"}:
        _finalize(config, archive, output_dir)
        return
    if command == "prepare-real-data":
        with logged_stage("Real-data training-input preparation", logger=LOGGER):
            _training_inputs(config, archive, output_dir, force=args.force)
        return

    with logged_stage("Real-data training-input preparation", logger=LOGGER):
        inputs = _training_inputs(config, archive, output_dir)
    if command == "select-nu":
        with logged_stage("Innovation-distribution selection", logger=LOGGER):
            select_nu(inputs, config, output_dir, workers=args.workers)
    elif command == "events":
        with logged_stage("Innovation-distribution selection lookup", logger=LOGGER):
            try:
                nu = _selected_nu(output_dir)
                LOGGER.info("Using cached Student-t degrees of freedom: %g", nu)
            except FileNotFoundError:
                nu = select_nu(inputs, config, output_dir, workers=args.workers)
        with logged_stage("Out-of-sample event study", logger=LOGGER):
            run_event_study(inputs, config, nu, output_dir, workers=args.workers)
    elif command in {"smoke", "full"}:
        with logged_stage("Simulation and continuous-monitoring study", logger=LOGGER):
            run_simulation(config, output_dir, build_dir, workers=args.workers)
        with logged_stage("Innovation-distribution selection", logger=LOGGER):
            nu = select_nu(inputs, config, output_dir, workers=args.workers)
        with logged_stage("Out-of-sample event study", logger=LOGGER):
            run_event_study(inputs, config, nu, output_dir, workers=args.workers)
        _finalize(config, archive, output_dir)
    else:
        raise AssertionError(command)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate BOCPD command-line arguments.

    Args:
        argv: Argument vector excluding the program name, or ``None`` to use
            the process arguments.

    Returns:
        Validated argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run deterministic changepoint-detection experiments."
    )
    parser.add_argument(
        "command",
        choices=(
            "prepare-real-data",
            "simulation",
            "select-nu",
            "events",
            "finalize",
            "validate",
            "smoke",
            "full",
            "clean",
        ),
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 2),
        help="independent streams or contracts evaluated concurrently",
    )
    parser.add_argument("--force", action="store_true", help="refit cached priors")
    args = parser.parse_args(argv)
    if args.command == "full":
        args.profile = "full"
    elif args.command == "smoke":
        args.profile = "smoke"
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    """Configure logging and execute the requested BOCPD command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    with logged_stage(
        f"Changepoint command '{args.command}' ({args.profile})", logger=LOGGER
    ):
        run_command(args)
