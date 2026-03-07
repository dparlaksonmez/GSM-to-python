"""
🧬 GSM Pipeline — Rich Interactive Command-Line Interface

Purpose:
    Beautiful, intuitive CLI for the GSM Bioinformatics Pipeline.
    Supports both direct commands and a guided interactive menu.

Usage:
    python -m gsm                  # Interactive menu (recommended)
    python -m gsm train            # Direct training command
    python -m gsm infer            # Clinical inference
    python -m gsm bundle-info      # Inspect a model bundle
    python -m gsm ui               # Launch Streamlit dashboard

Key Functions:
    - main(): Entry point — routes to interactive or direct mode
    - interactive_menu(): Guided menu for all operations
    - handle_train(): Train the GSM pipeline
    - handle_infer(): Run clinical inference
    - handle_bundle_info(): Display bundle metadata

Example Usage:
    python -m gsm
    python -m gsm train --test --iterations 5
    python -m gsm infer -b bundle.gsm.zip -p patients.csv
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Project root for resolving data paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


##### Rich Console Setup #####

def _get_console():
    """Lazy-import rich and return a configured Console."""
    from rich.console import Console
    return Console()


##### ASCII Banner #####

BANNER = r"""
[bold cyan]
   ╔══════════════════════════════════════════════════════════════╗
   ║                                                              ║
   ║    ██████╗ ███████╗███╗   ███╗                               ║
   ║   ██╔════╝ ██╔════╝████╗ ████║                               ║
   ║   ██║  ███╗███████╗██╔████╔██║                               ║
   ║   ██║   ██║╚════██║██║╚██╔╝██║                               ║
   ║   ╚██████╔╝███████║██║ ╚═╝ ██║                               ║
   ║    ╚═════╝ ╚══════╝╚═╝     ╚═╝                               ║
   ║                                                              ║
   ║   [bold white]Group  ·  Score  ·  Model[/bold white]                              ║
   ║   [dim]Bioinformatics Gene Expression Pipeline[/dim]                 ║
   ║                                                              ║
   ╚══════════════════════════════════════════════════════════════╝
[/bold cyan]"""

VERSION = "2.0.0"


##### Discovery Helpers #####

def _discover_datasets() -> list[Path]:
    """Find all CSV datasets in data/expression_data/."""
    data_dir = PROJECT_ROOT / "data" / "expression_data"
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.csv"))


def _discover_grouping_files() -> list[Path]:
    """Find all grouping files in data/grouping_data/."""
    group_dir = PROJECT_ROOT / "data" / "grouping_data"
    if not group_dir.exists():
        return []
    return sorted(group_dir.glob("*"))


def _discover_bundles() -> list[Path]:
    """Find all .gsm.zip bundles in output/."""
    output_dir = PROJECT_ROOT / "output"
    if not output_dir.exists():
        return []
    return sorted(output_dir.rglob("*.gsm.zip"))


def _discover_patient_files() -> list[Path]:
    """Find all CSV files in data/patient_data/."""
    patient_dir = PROJECT_ROOT / "data" / "patient_data"
    if not patient_dir.exists():
        return []
    return sorted(patient_dir.glob("*.csv"))


def _discover_output_runs() -> list[Path]:
    """Find all output run directories."""
    output_dir = PROJECT_ROOT / "output"
    if not output_dir.exists():
        return []
    runs = []
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and d.name.startswith(("gsm_", "gl_")):
            runs.append(d)
    return runs


##### Prompt Helpers #####

def _prompt_choice(console, prompt: str, choices: list[str],
                   descriptions: Optional[list[str]] = None,
                   allow_back: bool = True) -> Optional[int]:
    """Display a numbered menu and return the selected index (0-based), or None for back."""
    from rich.panel import Panel

    lines = []
    for i, choice in enumerate(choices, 1):
        desc = f"  [dim]{descriptions[i-1]}[/dim]" if descriptions else ""
        lines.append(f"  [bold cyan]{i}[/bold cyan])  {choice}{desc}")
    if allow_back:
        lines.append(f"  [bold red]0[/bold red])  ← Back")
    lines.append("")

    menu_text = "\n".join(lines)
    console.print(Panel(menu_text, title=f"[bold]{prompt}[/bold]",
                        border_style="cyan", padding=(1, 2)))

    while True:
        try:
            raw = console.input("[bold cyan]  ▶ Your choice: [/bold cyan]").strip()
            if not raw:
                continue
            num = int(raw)
            if allow_back and num == 0:
                return None
            if 1 <= num <= len(choices):
                return num - 1
            console.print(f"  [red]Please enter a number between "
                          f"{0 if allow_back else 1} and {len(choices)}[/red]")
        except ValueError:
            console.print("  [red]Please enter a number.[/red]")
        except (KeyboardInterrupt, EOFError):
            return None


def _prompt_text(console, prompt: str, default: str = "") -> str:
    """Prompt for free text input with an optional default."""
    suffix = f" [dim]({default})[/dim]" if default else ""
    try:
        raw = console.input(
            f"  [bold cyan]▶ {prompt}{suffix}: [/bold cyan]"
        ).strip()
        return raw or default
    except (KeyboardInterrupt, EOFError):
        return default


def _prompt_yes_no(console, prompt: str, default: bool = True) -> bool:
    """Prompt for a yes/no answer."""
    hint = "[Y/n]" if default else "[y/N]"
    try:
        raw = console.input(
            f"  [bold cyan]▶ {prompt} {hint}: [/bold cyan]"
        ).strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes", "1", "true")
    except (KeyboardInterrupt, EOFError):
        return default


##### Interactive Menu #####

def interactive_menu() -> None:
    """Launch the interactive guided menu."""
    console = _get_console()
    console.print(BANNER)
    console.print(
        f"  [dim]Version {VERSION} · Python {sys.version.split()[0]}[/dim]\n"
    )

    while True:
        choice = _prompt_choice(
            console,
            "What would you like to do?",
            [
                "🧬  Train Pipeline",
                "🏥  Clinical Inference",
                "🏥  Multi-Bundle Inference",
                "📦  Inspect Model Bundle",
                "📊  Launch Dashboard (Streamlit)",
                "⚡  Quick Test Run",
                "❓  Help & Documentation",
            ],
            [
                "Train classifiers on gene expression data",
                "Diagnose patients using a single trained model",
                "Combine multiple dataset models for robust diagnosis",
                "View metadata of a saved .gsm.zip bundle",
                "Open the web-based UI for results exploration",
                "Run a fast test with sample data (3 iterations)",
                "Show usage guide and available commands",
            ],
            allow_back=False,
        )

        if choice is None:
            _goodbye(console)
            return

        if choice == 0:
            _interactive_train(console)
        elif choice == 1:
            _interactive_infer(console)
        elif choice == 2:
            _interactive_multi_infer(console)
        elif choice == 3:
            _interactive_bundle_info(console)
        elif choice == 4:
            _launch_streamlit(console)
        elif choice == 5:
            _quick_test(console)
        elif choice == 6:
            _show_help(console)


def _goodbye(console) -> None:
    """Print goodbye message."""
    console.print(
        "\n  [bold cyan]👋 Goodbye! Happy researching.[/bold cyan]\n"
    )


##### Interactive Train #####

def _interactive_train(console) -> None:
    """Guided training flow."""
    from rich.table import Table

    console.print("\n[bold cyan]═══ 🧬 TRAIN PIPELINE ═══[/bold cyan]\n")

    # Step 1: Select dataset
    datasets = _discover_datasets()
    if not datasets:
        console.print("  [red]No datasets found in data/expression_data/[/red]")
        console.print(
            "  [dim]Place your expression CSV files there and try again.[/dim]\n"
        )
        return

    console.print("  [bold]Step 1/5 · Select Dataset[/bold]\n")
    names = [p.stem for p in datasets]
    sizes = []
    for d in datasets:
        try:
            n_rows = sum(1 for _ in open(d)) - 1
            sizes.append(f"{n_rows} samples")
        except Exception:
            sizes.append("")
    idx = _prompt_choice(console, "Available Datasets", names, sizes)
    if idx is None:
        return
    data_path = datasets[idx]

    # Step 2: Select grouping file
    groups = _discover_grouping_files()
    console.print("\n  [bold]Step 2/5 · Select Grouping File[/bold]\n")
    if len(groups) == 1:
        group_path = groups[0]
        console.print(
            f"  [green]✓[/green] Using: [bold]{group_path.name}[/bold]\n"
        )
    elif groups:
        gnames = [p.name for p in groups]
        gidx = _prompt_choice(console, "Available Grouping Files", gnames)
        if gidx is None:
            return
        group_path = groups[gidx]
    else:
        console.print(
            "  [red]No grouping files found in data/grouping_data/[/red]\n"
        )
        return

    # Step 3: Select model
    console.print("  [bold]Step 3/5 · Select Classifier[/bold]\n")
    models = [
        "RandomForest", "XGBoost", "DecisionTree", "SVM", "KNN", "MLP",
    ]
    model_desc = [
        "Best biological coherence (recommended)",
        "Fast alternative, 2.7× faster",
        "Simple, interpretable",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Neural Network",
    ]
    midx = _prompt_choice(console, "Classifiers", models, model_desc)
    if midx is None:
        return
    model_name = models[midx]

    # Step 4: Iterations & seed
    console.print("\n  [bold]Step 4/5 · Configure Run[/bold]\n")
    n_iter_str = _prompt_text(console, "Number of iterations", "100")
    n_iterations = int(n_iter_str) if n_iter_str.isdigit() else 100
    seed_str = _prompt_text(console, "Random seed", "44")
    seed = int(seed_str) if seed_str.isdigit() else 44

    # Step 5: Options
    console.print("\n  [bold]Step 5/5 · Options[/bold]\n")
    run_bio = _prompt_yes_no(
        console, "Run biological validation?", default=True
    )

    # Summary
    summary = Table(
        title="Training Configuration", border_style="cyan",
        show_header=False, padding=(0, 2),
    )
    summary.add_column("Setting", style="bold")
    summary.add_column("Value", style="green")
    summary.add_row("Dataset", data_path.stem)
    summary.add_row("Grouping", group_path.name)
    summary.add_row("Classifier", model_name)
    summary.add_row("Iterations", str(n_iterations))
    summary.add_row("Seed", str(seed))
    summary.add_row("Bio Validation", "Yes" if run_bio else "No")
    console.print()
    console.print(summary)
    console.print()

    if not _prompt_yes_no(console, "Start training?", default=True):
        console.print("  [dim]Training cancelled.[/dim]\n")
        return

    # Execute training
    _execute_train(
        console,
        data_path=data_path,
        group_path=group_path,
        model_name=model_name,
        n_iterations=n_iterations,
        seed=seed,
        run_bio=run_bio,
    )


##### Interactive Infer #####

def _interactive_infer(console) -> None:
    """Guided inference flow."""
    console.print("\n[bold cyan]═══ 🏥 CLINICAL INFERENCE ═══[/bold cyan]\n")

    # Step 1: Select bundle
    bundles = _discover_bundles()
    if not bundles:
        console.print("  [red]No model bundles found.[/red]")
        console.print(
            "  [dim]Train a pipeline first, then you'll find "
            ".gsm.zip files in output/[/dim]\n"
        )
        return

    console.print("  [bold]Step 1/2 · Select Model Bundle[/bold]\n")
    bundle_names = [
        f"{b.stem}  [dim]{b.parent.parent.name}[/dim]" for b in bundles
    ]
    bidx = _prompt_choice(console, "Available Bundles", bundle_names)
    if bidx is None:
        return
    bundle_path = bundles[bidx]

    # Step 2: Patient data
    console.print("\n  [bold]Step 2/2 · Patient Data[/bold]\n")
    console.print(
        "  [dim]Provide a CSV file where rows = samples, "
        "columns = gene IDs[/dim]"
    )
    console.print(
        "  [dim]Gene columns must match those used during training.[/dim]\n"
    )

    # Auto-discover patient files from data/patient_data/
    patient_files = _discover_patient_files()
    patients_path: Optional[Path] = None

    if patient_files:
        file_names = [f.name for f in patient_files]
        file_names.append("[dim]Enter a different path manually[/dim]")
        pidx = _prompt_choice(console, "Patient CSV Files", file_names)
        if pidx is None:
            return
        if pidx < len(patient_files):
            patients_path = patient_files[pidx]
        # else: fall through to manual entry below

    if patients_path is None:
        patients_path_str = _prompt_text(
            console, "Path to patient CSV file"
        )
        if not patients_path_str:
            console.print("  [red]No file provided.[/red]\n")
            return
        patients_path = Path(patients_path_str)

    if not patients_path.exists():
        console.print(f"  [red]File not found: {patients_path}[/red]\n")
        return

    # Optional: sample ID column
    sample_id_col = _prompt_text(
        console, "Sample ID column (leave blank to auto-number)"
    )

    console.print()
    if not _prompt_yes_no(console, "Run inference?", default=True):
        console.print("  [dim]Inference cancelled.[/dim]\n")
        return

    _execute_infer(
        console, bundle_path, patients_path,
        sample_id_col if sample_id_col else None,
    )


##### Interactive Multi-Bundle Infer #####

def _interactive_multi_infer(console) -> None:
    """Guided multi-bundle inference flow.

    Lets the user select multiple .gsm.zip bundles trained on
    different datasets and combine their predictions for more
    robust clinical inference.
    """
    console.print(
        "\n[bold cyan]═══ 🏥 MULTI-BUNDLE INFERENCE ═══[/bold cyan]\n"
    )

    # Step 1: Select multiple bundles
    bundles = _discover_bundles()
    if not bundles:
        console.print("  [red]No model bundles found.[/red]")
        console.print(
            "  [dim]Train pipelines on multiple datasets first.[/dim]\n"
        )
        return

    if len(bundles) < 2:
        console.print("  [yellow]Only 1 bundle found — need at least 2 for multi-bundle.[/yellow]")
        console.print(
            "  [dim]Use single-bundle inference instead, or train more datasets.[/dim]\n"
        )
        return

    console.print("  [bold]Step 1/2 · Select Model Bundles[/bold]\n")
    console.print(
        "  [dim]Select bundles one at a time. Enter 0 when done.[/dim]\n"
    )

    bundle_names = [
        f"{b.stem}  [dim]{b.parent.parent.name}[/dim]" for b in bundles
    ]
    selected_bundle_paths: list = []
    available_indices = list(range(len(bundles)))

    while True:
        remaining_names = [bundle_names[i] for i in available_indices]
        remaining_names.append("[bold green]✓ Done selecting[/bold green]")

        console.print(
            f"  [dim]Selected so far: {len(selected_bundle_paths)} bundles[/dim]"
        )
        bidx = _prompt_choice(
            console, "Available Bundles", remaining_names,
            allow_back=True,
        )
        if bidx is None:
            return

        # "Done selecting" option
        if bidx == len(remaining_names) - 1:
            break

        # Map back to original index
        original_idx = available_indices[bidx]
        selected_bundle_paths.append(bundles[original_idx])
        available_indices.remove(original_idx)
        console.print(
            f"  [green]Added: {bundles[original_idx].stem}[/green]"
        )

        if len(available_indices) == 0:
            break

    if len(selected_bundle_paths) < 2:
        console.print("  [red]Need at least 2 bundles for multi-bundle inference.[/red]\n")
        return

    console.print(
        f"\n  [bold]Selected {len(selected_bundle_paths)} bundles:[/bold]"
    )
    for bp in selected_bundle_paths:
        console.print(f"    • {bp.stem}")

    # Step 2: Patient data (same as single-bundle)
    console.print("\n  [bold]Step 2/2 · Patient Data[/bold]\n")
    console.print(
        "  [dim]Provide a CSV file where rows = samples, "
        "columns = gene IDs[/dim]\n"
    )

    patient_files = _discover_patient_files()
    patients_path: Optional[Path] = None

    if patient_files:
        file_names = [f.name for f in patient_files]
        file_names.append("[dim]Enter a different path manually[/dim]")
        pidx = _prompt_choice(console, "Patient CSV Files", file_names)
        if pidx is None:
            return
        if pidx < len(patient_files):
            patients_path = patient_files[pidx]

    if patients_path is None:
        patients_path_str = _prompt_text(
            console, "Path to patient CSV file"
        )
        if not patients_path_str:
            console.print("  [red]No file provided.[/red]\n")
            return
        patients_path = Path(patients_path_str)

    if not patients_path.exists():
        console.print(f"  [red]File not found: {patients_path}[/red]\n")
        return

    sample_id_col = _prompt_text(
        console, "Sample ID column (leave blank to auto-number)"
    )

    console.print()
    if not _prompt_yes_no(console, "Run multi-bundle inference?", default=True):
        console.print("  [dim]Inference cancelled.[/dim]\n")
        return

    _execute_multi_infer(
        console, selected_bundle_paths, patients_path,
        sample_id_col if sample_id_col else None,
    )


##### Interactive Bundle Info #####

def _interactive_bundle_info(console) -> None:
    """Guided bundle inspection flow."""
    console.print(
        "\n[bold cyan]═══ 📦 INSPECT MODEL BUNDLE ═══[/bold cyan]\n"
    )

    bundles = _discover_bundles()
    if not bundles:
        console.print("  [red]No model bundles found.[/red]")
        console.print(
            "  [dim]Train a pipeline first to create bundles.[/dim]\n"
        )
        return

    bundle_names = [
        f"{b.stem}  [dim]{b.parent.parent.name}[/dim]" for b in bundles
    ]
    bidx = _prompt_choice(console, "Available Bundles", bundle_names)
    if bidx is None:
        return

    _execute_bundle_info(console, bundles[bidx])


##### Launch Streamlit #####

def _launch_streamlit(console) -> None:
    """Launch the Streamlit dashboard."""
    console.print(
        "\n[bold cyan]═══ 📊 LAUNCHING DASHBOARD ═══[/bold cyan]\n"
    )
    app_path = PROJECT_ROOT / "src" / "ui" / "app.py"
    if not app_path.exists():
        console.print(
            "  [red]Streamlit app not found at src/ui/app.py[/red]\n"
        )
        return
    console.print(
        "  [dim]Starting Streamlit server... Press Ctrl+C to stop.[/dim]\n"
    )
    os.system(f"streamlit run {app_path}")


##### Quick Test #####

def _quick_test(console) -> None:
    """Run a quick test with sample data."""
    console.print("\n[bold cyan]═══ ⚡ QUICK TEST RUN ═══[/bold cyan]\n")

    data_path = PROJECT_ROOT / "data" / "test" / "test_expression_data.csv"
    group_path = PROJECT_ROOT / "data" / "test" / "test_grouping_data.csv"

    if not data_path.exists():
        console.print("  [red]Test data not found at data/test/[/red]\n")
        return

    console.print("  [dim]Running 3 iterations with test data...[/dim]\n")

    _execute_train(
        console,
        data_path=data_path,
        group_path=group_path,
        model_name="RandomForest",
        n_iterations=3,
        seed=44,
        run_bio=False,
        is_test=True,
    )


##### Help #####

def _show_help(console) -> None:
    """Display help and usage information."""
    from rich.panel import Panel
    from rich.table import Table

    console.print(
        "\n[bold cyan]═══ ❓ HELP & DOCUMENTATION ═══[/bold cyan]\n"
    )

    # Command reference
    t = Table(
        title="Command Reference", border_style="cyan", padding=(0, 2),
    )
    t.add_column("Command", style="bold cyan")
    t.add_column("Description")
    t.add_column("Example", style="dim")
    t.add_row("python -m gsm", "Interactive menu", "")
    t.add_row(
        "python -m gsm train", "Train pipeline",
        "--data GDS2545.csv --iterations 100",
    )
    t.add_row(
        "python -m gsm infer", "Clinical inference",
        "-b bundle.gsm.zip -p patients.csv",
    )
    t.add_row(
        "python -m gsm bundle-info", "Inspect bundle",
        "-b bundle.gsm.zip",
    )
    t.add_row("python -m gsm ui", "Streamlit dashboard", "")
    t.add_row(
        "python run_test.py", "Quick test (legacy)",
        "--iterations 5",
    )
    t.add_row(
        "python run_all_datasets.py", "Batch all datasets",
        "--iterations 50",
    )
    console.print(t)

    # Pipeline stages
    console.print()
    console.print(Panel(
        "[bold]Pipeline Stages:[/bold]\n\n"
        "  1. [cyan]Filter[/cyan]    →  Welch t-test (α=0.05) "
        "+ Benjamini-Hochberg FDR\n"
        "  2. [cyan]Group[/cyan]     →  Map genes to biological "
        "pathways (DisGeNET)\n"
        "  3. [cyan]Score[/cyan]     →  Rank groups by "
        "classification performance\n"
        "  4. [cyan]Model[/cyan]     →  Train final classifier "
        "on top gene groups\n"
        "  5. [cyan]Rank[/cyan]      →  Robust Rank Aggregation "
        "across iterations\n"
        "  6. [cyan]Validate[/cyan]  →  Enrichr + STRING-db "
        "+ DisGeNET queries\n"
        "  7. [cyan]Bundle[/cyan]    →  Package top models "
        "into .gsm.zip\n"
        "  8. [cyan]Infer[/cyan]     →  Diagnose new patients "
        "using model ensemble",
        title="[bold]GSM Pipeline Overview[/bold]",
        border_style="cyan",
    ))

    # Documentation links
    console.print()
    docs_t = Table(
        title="Documentation", border_style="cyan",
        show_header=False, padding=(0, 2),
    )
    docs_t.add_column("File", style="bold")
    docs_t.add_column("Description")
    docs_t.add_row("DOCS/RUNNING.md", "Detailed usage guide")
    docs_t.add_row("DOCS/DEVELOPMENT.md", "Developer guide")
    docs_t.add_row("DOCS/TROUBLESHOOTING.md", "Common issues & fixes")
    docs_t.add_row("README.md", "Project overview")
    console.print(docs_t)
    console.print()


##### Execution Engines #####

def _execute_train(
    console,
    data_path: Path,
    group_path: Path,
    model_name: str,
    n_iterations: int,
    seed: int,
    run_bio: bool,
    is_test: bool = False,
) -> None:
    """Execute the GSM training pipeline with rich progress display."""
    from rich.panel import Panel
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn,
    )
    import time

    from src.data_processing.data_loader import (
        load_input_file, load_group_file,
    )
    from src.workflows.GSM_workflow import gsm_run
    from src.workflows.GSM_workflow_config import (
        MAIN_DATA_FILE_SEPARATOR, GROUPING_FILE_SEPARATOR,
        TRAIN_TEST_SPLIT_RATIO, SCORING_MODEL,
        LABEL_COLUMN_NAME, CLASS_LABELS_POSITIVE, CLASS_LABELS_NEGATIVE,
        GENE_COLUMN_NAME, GROUP_COLUMN_NAME, NORMALIZATION_METHOD,
        BIOLOGICAL_VALIDATION_TOP_GENES, DISGENET_API_KEY,
    )

    console.print()
    console.print(Panel(
        f"  Dataset:     [bold]{data_path.stem}[/bold]\n"
        f"  Grouping:    [bold]{group_path.name}[/bold]\n"
        f"  Classifier:  [bold]{model_name}[/bold]\n"
        f"  Iterations:  [bold]{n_iterations}[/bold]\n"
        f"  Seed:        [bold]{seed}[/bold]",
        title="[bold cyan]🧬 Starting Training[/bold cyan]",
        border_style="green",
    ))

    # Load data with spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Loading expression data...", total=None,
        )
        input_data = load_input_file(
            data_path, separator=MAIN_DATA_FILE_SEPARATOR,
        )
        progress.update(task, description="Loading grouping data...")
        group_data = load_group_file(
            group_path, separator=GROUPING_FILE_SEPARATOR,
        )
        progress.update(task, description="Data loaded!")

    console.print(
        f"  [green]✓[/green] Expression data: "
        f"[bold]{input_data.shape[0]}[/bold] samples × "
        f"[bold]{input_data.shape[1]}[/bold] features"
    )
    console.print(
        f"  [green]✓[/green] Grouping data:   "
        f"[bold]{group_data.shape[0]}[/bold] gene-group mappings\n"
    )

    # Run pipeline
    t0 = time.time()
    console.print("  [bold]Running GSM pipeline...[/bold]")
    console.print(
        "  [dim]Progress details in log output below ↓[/dim]\n"
    )

    try:
        output_path = gsm_run(
            input_data,
            group_data,
            n_iterations=n_iterations,
            model_name=model_name,
            scoring_model=SCORING_MODEL,
            initial_seed=seed,
            sample_ratio=TRAIN_TEST_SPLIT_RATIO,
            label_column=LABEL_COLUMN_NAME,
            positive_class_label=CLASS_LABELS_POSITIVE,
            negative_class_label=CLASS_LABELS_NEGATIVE,
            gene_column=GENE_COLUMN_NAME,
            group_column=GROUP_COLUMN_NAME,
            normalization_method=NORMALIZATION_METHOD,
            run_biological_validation_flag=run_bio,
            biological_validation_top_genes=(
                BIOLOGICAL_VALIDATION_TOP_GENES
            ),
            disgenet_api_key=DISGENET_API_KEY,
            input_data_name=data_path.stem,
            group_data_name=group_path.stem,
        )
    except Exception as e:
        console.print(
            f"\n  [bold red]✗ Pipeline failed: {e}[/bold red]\n"
        )
        return

    elapsed = time.time() - t0

    # Success display
    console.print()
    result_lines = [
        f"  Status:      [bold green]SUCCESS[/bold green]",
        f"  Time:        [bold]{elapsed:.1f}s[/bold]",
        f"  Output:      [bold]{output_path}[/bold]",
    ]

    # Check for bundle
    bundles_dir = output_path / "bundles" if output_path else None
    if bundles_dir and bundles_dir.exists():
        bundle_files = list(bundles_dir.glob("*.gsm.zip"))
        if bundle_files:
            result_lines.append(
                f"  Bundle:      [bold]{bundle_files[0].name}[/bold]"
            )

    console.print(Panel(
        "\n".join(result_lines),
        title="[bold green]🎉 Training Complete[/bold green]",
        border_style="green",
    ))

    # Next steps
    if bundles_dir and bundles_dir.exists():
        bundle_files = list(bundles_dir.glob("*.gsm.zip"))
        if bundle_files:
            console.print()
            console.print(Panel(
                f"  To diagnose patients:\n"
                f"  [bold cyan]python -m gsm infer "
                f"-b {bundle_files[0]} -p patients.csv[/bold cyan]\n\n"
                f"  To inspect the bundle:\n"
                f"  [bold cyan]python -m gsm bundle-info "
                f"-b {bundle_files[0]}[/bold cyan]",
                title="[bold]Next Steps[/bold]",
                border_style="dim",
            ))
    console.print()


def _execute_infer(
    console,
    bundle_path: Path,
    patients_path: Path,
    sample_id_column: Optional[str] = None,
) -> None:
    """Execute clinical inference with rich result display."""
    import pandas as pd
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from src.inference.model_bundle import load_bundle
    from src.inference.inference_engine import infer
    from src.inference.clinical_report import (
        generate_clinical_report,
        save_clinical_report,
    )
    from src.utils.logger import setup_logger

    output_dir = bundle_path.parent.parent / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        str(output_dir / "inference.log"),
        logger_name="GSM_inference",
    )

    console.print()

    # Load with spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Loading model bundle...", total=None,
        )
        bundle = load_bundle(bundle_path, logger=logger)

        progress.update(
            task, description="Loading patient data...",
        )
        patient_data = pd.read_csv(
            patients_path, sep=None, engine="python",
        )
        progress.update(
            task, description="Running ensemble inference...",
        )
        summary = infer(
            bundle, patient_data, logger=logger,
            sample_id_column=sample_id_column,
        )

    # Bundle info compact
    console.print(Panel(
        f"  Models:   [bold]{bundle.metadata.n_models_saved}[/bold] "
        f"in ensemble\n"
        f"  Features: [bold]{bundle.metadata.n_features}[/bold] genes\n"
        f"  F1:       [bold]{bundle.metadata.ensemble_f1_mean:.4f}[/bold]"
        f" ± {bundle.metadata.ensemble_f1_std:.4f}\n"
        f"  Patients: [bold]{len(patient_data)}[/bold] samples",
        title="[bold cyan]🏥 Inference Summary[/bold cyan]",
        border_style="cyan",
    ))

    # Results table
    results_table = Table(
        title="Patient Predictions",
        border_style="cyan",
        show_lines=True,
        padding=(0, 1),
    )
    results_table.add_column("Sample", style="bold")
    results_table.add_column("Prediction", justify="center")
    results_table.add_column("Probability", justify="center")
    results_table.add_column("Risk", justify="center")
    results_table.add_column("Confidence", justify="center")
    results_table.add_column("Agreement", justify="center")
    results_table.add_column("Top Contributing Genes", max_width=40)

    for result in summary.results:
        # Color-code risk
        risk = result.risk_level
        if risk == "HIGH":
            risk_styled = "[bold red]HIGH[/bold red]"
        elif risk == "MEDIUM":
            risk_styled = "[bold yellow]MEDIUM[/bold yellow]"
        else:
            risk_styled = "[bold green]LOW[/bold green]"

        # Color-code prediction
        pred_label = result.predicted_label
        if pred_label.lower() in (
            "pos", "positive", "1", "true", "disease",
        ):
            pred_styled = f"[bold red]{pred_label}[/bold red]"
        else:
            pred_styled = f"[bold green]{pred_label}[/bold green]"

        # Format confidence
        conf = result.confidence
        if conf >= 0.9:
            conf_styled = f"[bold green]{conf:.1%}[/bold green]"
        elif conf >= 0.7:
            conf_styled = f"[yellow]{conf:.1%}[/yellow]"
        else:
            conf_styled = f"[red]{conf:.1%}[/red]"

        # Top genes
        top_genes = ", ".join(
            list(result.top_contributing_genes.keys())[:5]
        )

        # Mean probability from individual model probs
        mean_prob = (
            sum(result.individual_probabilities)
            / len(result.individual_probabilities)
            if result.individual_probabilities else 0.0
        )

        results_table.add_row(
            str(result.sample_id),
            pred_styled,
            f"{mean_prob:.3f}",
            risk_styled,
            conf_styled,
            f"{result.agreement_ratio:.1%}",
            top_genes,
        )

    console.print()
    console.print(results_table)

    # Save reports
    save_clinical_report(summary, output_dir, logger)

    console.print()
    console.print(Panel(
        f"  Reports saved to: [bold]{output_dir}[/bold]\n"
        f"  Files: clinical_report.txt, clinical_report.csv",
        title="[bold green]📁 Reports Saved[/bold green]",
        border_style="green",
    ))
    console.print()


def _execute_multi_infer(
    console,
    bundle_paths: list[Path],
    patients_path: Path,
    sample_id_column: Optional[str] = None,
) -> None:
    """Execute multi-bundle inference with rich result display."""
    import pandas as pd
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from src.inference.model_bundle import load_bundle
    from src.inference.inference_engine import multi_infer
    from src.inference.clinical_report import save_multi_bundle_report
    from src.utils.logger import setup_logger

    # Use first bundle's parent for output
    output_dir = bundle_paths[0].parent.parent / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        str(output_dir / "multi_inference.log"),
        logger_name="GSM_multi_inference",
    )

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Loading model bundles...", total=None,
        )
        loaded_bundles = []
        for i, bp in enumerate(bundle_paths):
            progress.update(
                task,
                description=f"Loading bundle {i+1}/{len(bundle_paths)}...",
            )
            loaded_bundles.append(load_bundle(bp, logger=logger))

        progress.update(
            task, description="Loading patient data...",
        )
        patient_data = pd.read_csv(
            patients_path, sep=None, engine="python",
        )

        progress.update(
            task, description="Running multi-bundle inference...",
        )
        summary = multi_infer(
            loaded_bundles, patient_data,
            logger=logger,
            sample_id_column=sample_id_column,
        )

    # Summary panel
    bundle_info_lines = []
    for b in loaded_bundles:
        m = b.metadata
        bundle_info_lines.append(
            f"  • {m.dataset_name}: {m.n_models_saved} models, "
            f"{m.n_features} features, F1={m.ensemble_f1_mean:.3f}"
        )

    console.print(Panel(
        f"  Bundles:  [bold]{summary.n_bundles}[/bold]\n"
        + "\n".join(bundle_info_lines) + "\n"
        f"  Patients: [bold]{summary.n_samples}[/bold] samples",
        title="[bold cyan]🏥 Multi-Bundle Inference[/bold cyan]",
        border_style="cyan",
    ))

    # Results table
    results_table = Table(
        title="Consensus Predictions",
        border_style="cyan",
        show_lines=True,
        padding=(0, 1),
    )
    results_table.add_column("Sample", style="bold")
    results_table.add_column("Consensus", justify="center")
    results_table.add_column("Confidence", justify="center")
    results_table.add_column("Risk", justify="center")
    results_table.add_column("Bundles +", justify="center")
    results_table.add_column("Agreement", justify="center")
    results_table.add_column("Top Genes (merged)", max_width=40)

    for r in summary.results:
        risk = r.risk_level
        if risk == "HIGH":
            risk_styled = "[bold red]HIGH[/bold red]"
        elif risk == "MEDIUM":
            risk_styled = "[bold yellow]MEDIUM[/bold yellow]"
        else:
            risk_styled = "[bold green]LOW[/bold green]"

        pred = r.predicted_class
        if pred == "positive":
            pred_styled = "[bold red]positive[/bold red]"
        else:
            pred_styled = "[bold green]negative[/bold green]"

        conf = r.consensus_confidence
        if conf >= 0.9:
            conf_styled = f"[bold green]{conf:.1%}[/bold green]"
        elif conf >= 0.7:
            conf_styled = f"[yellow]{conf:.1%}[/yellow]"
        else:
            conf_styled = f"[red]{conf:.1%}[/red]"

        top_genes = ", ".join(
            list(r.top_contributing_genes.keys())[:5]
        )

        results_table.add_row(
            str(r.sample_id),
            pred_styled,
            conf_styled,
            risk_styled,
            f"{r.n_bundles_positive}/{r.n_bundles_total}",
            f"{r.agreement_ratio:.0%}",
            top_genes,
        )

    console.print()
    console.print(results_table)

    # Save reports
    save_multi_bundle_report(summary, output_dir, logger)

    console.print()
    console.print(Panel(
        f"  Reports saved to: [bold]{output_dir}[/bold]\n"
        f"  Files: multi_bundle_report.txt, multi_bundle_report.xlsx",
        title="[bold green]📁 Reports Saved[/bold green]",
        border_style="green",
    ))
    console.print()


def _execute_bundle_info(console, bundle_path: Path) -> None:
    """Display bundle info with rich formatting."""
    from rich.panel import Panel
    from src.inference.model_bundle import bundle_info

    info_text = bundle_info(bundle_path)

    console.print()
    console.print(Panel(
        info_text,
        title=f"[bold cyan]📦 {bundle_path.name}[/bold cyan]",
        border_style="cyan",
    ))
    console.print()


##### Argparse CLI (Direct Commands) #####

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for non-interactive use."""
    parser = argparse.ArgumentParser(
        prog="gsm",
        description=(
            "🧬 GSM Bioinformatics Pipeline — "
            "Train classifiers & diagnose patients"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Interactive mode (no arguments):\n"
            "  python -m gsm\n\n"
            "Direct commands:\n"
            "  python -m gsm train --data data/expression_data/GDS2545.csv "
            "--iterations 100\n"
            "  python -m gsm infer -b bundle.gsm.zip -p patients.csv\n"
            "  python -m gsm multi-infer -b b1.gsm.zip b2.gsm.zip "
            "-p patients.csv\n"
            "  python -m gsm bundle-info -b bundle.gsm.zip\n"
            "  python -m gsm ui\n"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands",
    )

    # ── train ──
    train_p = subparsers.add_parser(
        "train", help="Train the GSM pipeline",
    )
    train_p.add_argument(
        "--data", type=str, default=None,
        help="Path to expression data CSV",
    )
    train_p.add_argument(
        "--groups", type=str, default=None,
        help="Path to grouping data file",
    )
    train_p.add_argument(
        "--iterations", "-n", type=int, default=None,
        help="Number of iterations (default: config)",
    )
    train_p.add_argument(
        "--model", type=str, default=None,
        choices=[
            "RandomForest", "XGBoost", "DecisionTree",
            "SVM", "KNN", "MLP",
        ],
        help="Classifier to use",
    )
    train_p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: 44)",
    )
    train_p.add_argument(
        "--test", action="store_true",
        help="Use test data for a quick run",
    )
    train_p.add_argument(
        "--no-bio-validation", action="store_true",
        help="Skip biological validation",
    )

    # ── infer ──
    infer_p = subparsers.add_parser(
        "infer", help="Clinical inference on patient data",
    )
    infer_p.add_argument(
        "--bundle", "-b", type=str, required=True,
        help="Path to .gsm.zip bundle",
    )
    infer_p.add_argument(
        "--patients", "-p", type=str, required=True,
        help="Path to patient expression CSV",
    )
    infer_p.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for reports",
    )
    infer_p.add_argument(
        "--sample-id-column", type=str, default=None,
        help="Column with sample IDs",
    )

    # ── bundle-info ──
    info_p = subparsers.add_parser(
        "bundle-info", help="Inspect a model bundle",
    )
    info_p.add_argument(
        "--bundle", "-b", type=str, required=True,
        help="Path to .gsm.zip bundle",
    )

    # ── multi-infer ──
    multi_p = subparsers.add_parser(
        "multi-infer",
        help="Multi-bundle inference (combine multiple datasets)",
    )
    multi_p.add_argument(
        "--bundles", "-b", type=str, nargs="+", required=True,
        help="Paths to .gsm.zip bundles (2 or more)",
    )
    multi_p.add_argument(
        "--patients", "-p", type=str, required=True,
        help="Path to patient expression CSV",
    )
    multi_p.add_argument(
        "--sample-id-column", type=str, default=None,
        help="Column with sample IDs",
    )

    # ── ui ──
    subparsers.add_parser("ui", help="Launch Streamlit dashboard")

    return parser


def _handle_train_args(args: argparse.Namespace) -> None:
    """Handle direct train command."""
    from src.workflows.GSM_workflow_config import (
        INPUT_EXPRESSION_DATA, INPUT_GROUP_DATA,
        NUMBER_OF_ITERATIONS, RANDOM_SEED, MODEL_NAME,
        RUN_BIOLOGICAL_VALIDATION,
    )

    console = _get_console()
    console.print(BANNER)

    if args.test:
        data_path = (
            PROJECT_ROOT / "data" / "test" / "test_expression_data.csv"
        )
        group_path = (
            PROJECT_ROOT / "data" / "test" / "test_grouping_data.csv"
        )
        n_iterations = args.iterations or 3
    else:
        data_path = (
            Path(args.data) if args.data
            else PROJECT_ROOT / INPUT_EXPRESSION_DATA
        )
        group_path = (
            Path(args.groups) if args.groups
            else PROJECT_ROOT / INPUT_GROUP_DATA
        )
        n_iterations = args.iterations or NUMBER_OF_ITERATIONS

    _execute_train(
        console,
        data_path=data_path,
        group_path=group_path,
        model_name=args.model or MODEL_NAME,
        n_iterations=n_iterations,
        seed=args.seed or RANDOM_SEED,
        run_bio=(
            not args.no_bio_validation and RUN_BIOLOGICAL_VALIDATION
        ),
        is_test=args.test,
    )


def _handle_infer_args(args: argparse.Namespace) -> None:
    """Handle direct infer command."""
    console = _get_console()
    console.print(BANNER)

    bundle_path = Path(args.bundle)
    patients_path = Path(args.patients)

    if not bundle_path.exists():
        console.print(
            f"  [bold red]✗ Bundle not found: {bundle_path}[/bold red]\n"
        )
        sys.exit(1)
    if not patients_path.exists():
        console.print(
            f"  [bold red]✗ Patient file not found: "
            f"{patients_path}[/bold red]\n"
        )
        sys.exit(1)

    _execute_infer(
        console, bundle_path, patients_path, args.sample_id_column,
    )


def _handle_bundle_info_args(args: argparse.Namespace) -> None:
    """Handle direct bundle-info command."""
    console = _get_console()
    console.print(BANNER)

    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        console.print(
            f"  [bold red]✗ Bundle not found: "
            f"{bundle_path}[/bold red]\n"
        )
        sys.exit(1)

    _execute_bundle_info(console, bundle_path)


def _handle_ui_args(args: argparse.Namespace) -> None:
    """Handle direct ui command."""
    console = _get_console()
    console.print(BANNER)
    _launch_streamlit(console)


def _handle_multi_infer_args(args: argparse.Namespace) -> None:
    """Handle direct multi-infer command."""
    console = _get_console()
    console.print(BANNER)

    bundle_paths = [Path(b) for b in args.bundles]
    patients_path = Path(args.patients)

    for bp in bundle_paths:
        if not bp.exists():
            console.print(
                f"  [bold red]✗ Bundle not found: {bp}[/bold red]\n"
            )
            sys.exit(1)

    if not patients_path.exists():
        console.print(
            f"  [bold red]✗ Patient file not found: "
            f"{patients_path}[/bold red]\n"
        )
        sys.exit(1)

    if len(bundle_paths) < 2:
        console.print(
            "  [bold red]✗ Need at least 2 bundles for "
            "multi-bundle inference.[/bold red]\n"
        )
        sys.exit(1)

    _execute_multi_infer(
        console, bundle_paths, patients_path,
        args.sample_id_column,
    )


##### Main Entry Point #####

def main() -> None:
    """Main CLI entry point.

    No arguments → interactive menu.
    With subcommand → direct dispatch.
    """
    # If no arguments, launch interactive mode
    if len(sys.argv) <= 1 or (
        len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help")
    ):
        if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
            # Show help via argparse for --help flag
            parser = _build_parser()
            console = _get_console()
            console.print(BANNER)
            parser.print_help()
            return
        # No args → interactive
        interactive_menu()
        return

    # Parse and dispatch
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        interactive_menu()
    elif args.command == "train":
        _handle_train_args(args)
    elif args.command == "infer":
        _handle_infer_args(args)
    elif args.command == "multi-infer":
        _handle_multi_infer_args(args)
    elif args.command == "bundle-info":
        _handle_bundle_info_args(args)
    elif args.command == "ui":
        _handle_ui_args(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
