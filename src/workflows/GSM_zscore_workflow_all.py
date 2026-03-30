import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt"}
DEFAULT_THRESHOLDS = (0.05, 0.01)
DEFAULT_MIN_CPUS_PER_WORKER = 4

CHILD_RUNNER = """
from pathlib import Path
import sys

import src.workflows.GSM_zscore_workflow_config as config

config.INPUT_EXPRESSION_DATA = sys.argv[1]
config.Q_VALUE_THRESHOLD = float(sys.argv[2])
config.OUTPUT_DIR = Path(sys.argv[3])

import src.workflows.GSM_zscore_workflow as workflow
workflow.main()
"""


@dataclass(frozen=True)
class DatasetJob:
    file_path: Path
    rel_path: str
    threshold: float


@dataclass(frozen=True)
class ExecutionPlan:
    total_cpus: int
    max_workers: int
    cpus_per_worker: int
    min_cpus_per_worker: int


@dataclass
class JobResult:
    dataset_name: str
    threshold: float
    success: bool
    returncode: int
    duration_seconds: float
    stdout_log: str
    stderr_log: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GSM z-score workflow across all datasets with controlled parallelism."
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Q-value thresholds to run sequentially.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of datasets to process at the same time.",
    )
    parser.add_argument(
        "--cpus-per-worker",
        type=int,
        default=None,
        help="CPU budget for each dataset process. Overrides auto allocation when set.",
    )
    parser.add_argument(
        "--min-cpus-per-worker",
        type=int,
        default=DEFAULT_MIN_CPUS_PER_WORKER,
        help="Auto mode keeps at least this many CPUs per dataset process when possible.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset filenames to run (e.g. GDS1962.csv GDS2545.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without starting any subprocesses.",
    )
    return parser.parse_args()


def detect_available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def discover_datasets(expression_data_dir: Path, requested_names: Sequence[str] | None) -> list[tuple[Path, str]]:
    dataset_files = sorted(
        (
            (file_path, f"data/expression_data/{file_path.name}")
            for file_path in expression_data_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda item: item[0].name,
    )

    if not requested_names:
        return dataset_files

    requested_set = set(requested_names)
    filtered = [item for item in dataset_files if item[0].name in requested_set]
    missing = sorted(requested_set - {item[0].name for item in filtered})
    if missing:
        raise ValueError(f"Requested datasets not found: {', '.join(missing)}")
    return filtered


def build_execution_plan(
    dataset_count: int,
    total_cpus: int,
    max_workers: int | None,
    cpus_per_worker: int | None,
    min_cpus_per_worker: int,
) -> ExecutionPlan:
    if dataset_count <= 0:
        return ExecutionPlan(total_cpus=total_cpus, max_workers=0, cpus_per_worker=0, min_cpus_per_worker=0)

    if max_workers is not None and max_workers < 1:
        raise ValueError("--max-workers must be at least 1")
    if cpus_per_worker is not None and cpus_per_worker < 1:
        raise ValueError("--cpus-per-worker must be at least 1")
    if min_cpus_per_worker < 1:
        raise ValueError("--min-cpus-per-worker must be at least 1")

    if max_workers is not None and cpus_per_worker is not None:
        resolved_workers = min(dataset_count, max_workers)
        resolved_cpus = cpus_per_worker
    elif max_workers is not None:
        resolved_workers = min(dataset_count, max_workers, total_cpus)
        resolved_cpus = max(1, total_cpus // resolved_workers)
    elif cpus_per_worker is not None:
        resolved_cpus = cpus_per_worker
        resolved_workers = min(dataset_count, max(1, total_cpus // resolved_cpus))
    else:
        target_min_cpus = min(total_cpus, min_cpus_per_worker)
        resolved_workers = min(dataset_count, max(1, total_cpus // target_min_cpus))
        resolved_cpus = max(1, total_cpus // resolved_workers)

    return ExecutionPlan(
        total_cpus=total_cpus,
        max_workers=max(1, resolved_workers),
        cpus_per_worker=max(1, resolved_cpus),
        min_cpus_per_worker=min_cpus_per_worker,
    )


def format_threshold(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def create_child_env(cpus_per_worker: int) -> dict[str, str]:
    env = os.environ.copy()
    env["GSM_INNER_N_JOBS"] = str(cpus_per_worker)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["JOBLIB_TEMP_FOLDER"] = "/tmp"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_single_dataset(
    job: DatasetJob,
    *,
    project_root: Path,
    workflow_output_root: Path,
    log_dir: Path,
    cpus_per_worker: int,
) -> JobResult:
    stdout_log = log_dir / f"{job.file_path.stem}.stdout.log"
    stderr_log = log_dir / f"{job.file_path.stem}.stderr.log"
    env = create_child_env(cpus_per_worker)

    command = [
        sys.executable,
        "-c",
        CHILD_RUNNER,
        job.rel_path,
        str(job.threshold),
        str(workflow_output_root),
    ]

    start_time = time.perf_counter()
    with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    duration_seconds = time.perf_counter() - start_time

    return JobResult(
        dataset_name=job.file_path.name,
        threshold=job.threshold,
        success=completed.returncode == 0,
        returncode=completed.returncode,
        duration_seconds=duration_seconds,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
    )


def run_threshold_batch(
    dataset_files: Sequence[tuple[Path, str]],
    threshold: float,
    *,
    project_root: Path,
    batch_root: Path,
    plan: ExecutionPlan,
) -> list[JobResult]:
    threshold_root = batch_root / f"q_{format_threshold(threshold)}"
    workflow_output_root = threshold_root / "workflow_output"
    log_dir = threshold_root / "launcher_logs"
    workflow_output_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = [DatasetJob(file_path=file_path, rel_path=rel_path, threshold=threshold) for file_path, rel_path in dataset_files]
    results: list[JobResult] = []

    print(f"\n{'#' * 88}")
    print(f"Q_VALUE_THRESHOLD = {threshold}")
    print(f"[plan] total_cpus={plan.total_cpus} | parallel_datasets={plan.max_workers} | cpus_per_dataset={plan.cpus_per_worker}")
    print(f"[plan] workflow_output_root={workflow_output_root}")
    print(f"[plan] logs={log_dir}")
    print(f"{'#' * 88}")

    with ThreadPoolExecutor(max_workers=plan.max_workers) as executor:
        future_to_job = {
            executor.submit(
                run_single_dataset,
                job,
                project_root=project_root,
                workflow_output_root=workflow_output_root,
                log_dir=log_dir,
                cpus_per_worker=plan.cpus_per_worker,
            ): job
            for job in jobs
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            result = future.result()
            status = "done" if result.success else "error"
            print(
                f"[{status}] {job.file_path.name} | q={threshold} | "
                f"duration={result.duration_seconds:.1f}s | returncode={result.returncode}"
            )
            results.append(result)

    results.sort(key=lambda item: item.dataset_name)
    summary_path = threshold_root / "batch_summary.json"
    summary_path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")

    successful = sum(1 for result in results if result.success)
    failed = len(results) - successful
    print(f"\n[summary] q={threshold} | success={successful}/{len(results)} | failed={failed}/{len(results)}")
    if failed:
        for result in results:
            if not result.success:
                print(f"[summary] failed={result.dataset_name} | stderr={result.stderr_log}")

    return results


def flatten_results(results_by_threshold: Iterable[list[JobResult]]) -> list[JobResult]:
    flat: list[JobResult] = []
    for batch in results_by_threshold:
        flat.extend(batch)
    return flat


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    expression_data_dir = project_root / "data" / "expression_data"

    if not expression_data_dir.exists():
        raise FileNotFoundError(f"Expression data directory does not exist: {expression_data_dir}")

    dataset_files = discover_datasets(expression_data_dir, args.datasets)
    if not dataset_files:
        raise FileNotFoundError(f"No dataset files found in {expression_data_dir}")

    total_cpus = detect_available_cpus()
    plan = build_execution_plan(
        dataset_count=len(dataset_files),
        total_cpus=total_cpus,
        max_workers=args.max_workers,
        cpus_per_worker=args.cpus_per_worker,
        min_cpus_per_worker=args.min_cpus_per_worker,
    )

    batch_root = project_root / "output" / "batch_runs" / time.strftime("%Y_%m_%d-%H_%M_%S")
    batch_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] project_root={project_root}")
    print(f"[info] datasets={len(dataset_files)}")
    print(f"[info] detected_cpus={total_cpus}")
    print(f"[info] plan={plan}")
    print(f"[info] batch_root={batch_root}")

    if args.dry_run:
        print("[info] Dry run requested, exiting without launching subprocesses.")
        return

    all_results: list[list[JobResult]] = []
    for threshold in args.thresholds:
        threshold_results = run_threshold_batch(
            dataset_files,
            threshold,
            project_root=project_root,
            batch_root=batch_root,
            plan=plan,
        )
        all_results.append(threshold_results)

    final_results = flatten_results(all_results)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batch_root": str(batch_root),
        "plan": asdict(plan),
        "thresholds": args.thresholds,
        "results": [asdict(result) for result in final_results],
    }
    (batch_root / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_failed = sum(1 for result in final_results if not result.success)
    print(f"\n[final] total_runs={len(final_results)} | failed={total_failed} | manifest={batch_root / 'batch_manifest.json'}")


if __name__ == "__main__":
    main()
