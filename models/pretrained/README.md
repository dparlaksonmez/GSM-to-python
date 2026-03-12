# Pre-trained Models

This folder stores `.gsm.zip` model bundles that ship with the
GSM pipeline so users can run inference **without training from scratch**.

## Available Publication Bundles

These bundles were trained using the main GSM pipeline (100 iterations,
Random Forest, seed 44) for the publication. Each contains the top-10
models by F1 score.

| Bundle | Disease | Samples | Features |
|--------|---------|---------|----------|
| `bundle_GDS1962_publication_RF.gsm.zip` | Glioblastoma | 180 | 54 613 |
| `bundle_GDS2545_publication_RF.gsm.zip` | Prostate Cancer | 171 | 12 580 |
| `bundle_GDS2547_publication_RF.gsm.zip` | Prostate Cancer (Lapointe) | 164 | 12 646 |
| `bundle_GDS2771_publication_RF.gsm.zip` | Lung Cancer | 192 | 22 215 |
| `bundle_GDS3257_publication_RF.gsm.zip` | Acute Myeloid Leukemia | 107 | 22 225 |
| `bundle_GDS3837_publication_RF.gsm.zip` | Colorectal Cancer | 120 | 30 622 |
| `bundle_GDS5499_publication_RF.gsm.zip` | Pancreatic Cancer | 140 | 48 803 |

## How to add a pre-trained model

1. Train the pipeline on a dataset:
   ```bash
   python -m src.cli train --dataset GDS1962
   ```
2. Copy the resulting bundle into this folder:
   ```bash
   cp output/gsm_<run>/bundles/*.gsm.zip models/pretrained/
   ```
3. Commit the bundle (use Git LFS for large files):
   ```bash
   git lfs track "models/pretrained/*.gsm.zip"
   git add models/pretrained/
   git commit -m "Add pre-trained model for GDS1962"
   ```

## How bundles are discovered

Both the CLI and the Streamlit app scan **two locations**:

| Location              | Purpose                        |
|-----------------------|--------------------------------|
| `output/*/bundles/`   | Locally trained models         |
| `models/pretrained/`  | Shipped / pre-trained models   |

Pre-trained bundles appear automatically in the inference menus
with a `[pretrained]` tag.

## HuggingFace Spaces

When deploying to HuggingFace Spaces, include this folder in
the repository. The Streamlit app detects it at startup and
offers inference-only mode when no local training data exists.

## Bundle format

Each `.gsm.zip` is a self-contained archive containing:
- Trained model(s) (pickle)
- Feature list and metadata (JSON)
- Group definitions (JSON)
- Pipeline configuration snapshot

See `src/inference/model_bundle.py` for the bundle API.
