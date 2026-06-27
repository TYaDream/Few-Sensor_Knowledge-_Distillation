#Gas Sensor Distillation

Code release for gas sensor concentration regression experiments on Zhang's dataset. The repository contains the training/evaluation code, experiment launch scripts, and table-generation utilities for teacher-student knowledge distillation baselines and FSKD variants.

> Data files, model checkpoints, and generated experiment outputs are intentionally not included. Put your dataset and checkpoints in local folders and pass their paths through command-line arguments.

## Repository Structure

```text
.
├── src/
│   ├── data.py                         # Dataset loading and preprocessing
│   ├── model.py                        # Teacher/student model definitions and KD losses
│   ├── run_zhang_paper_experiments.py  # Main baselines, FSKD, ablation, analysis runner
│   └── run_zhang_top50.py              # Top-k reliable-sample FSKD runner
├── scripts/
│   ├── run_paper_baselines_8tasks.ps1
│   ├── run_gamma_temperature_parallel.ps1
│   ├── run_top50_grid.ps1
│   ├── run_top50_smallgamma_grid.ps1
│   ├── run_best_task_ablation.ps1
│   ├── summarize_paper_baselines.py
│   └── make_paper_tables_zhang.py
├── results/                            # Empty placeholder for local outputs
├── requirements.txt
└── README.md
```

## Environment

Recommended:

- Python 3.9+
- CUDA-capable GPU for full 1000-epoch experiments
- Windows PowerShell if using the provided `.ps1` launch scripts

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install PyTorch first from the official selector, then run `pip install -r requirements.txt`.

## Data Preparation

The code expects `--data-root` to point to the preprocessed Zhang gas sensor dataset directory. The original dataset is not included in this repository.

Example local layout:

```text
/path/to/zhang_dataset/
├── ... dataset files used by src/data.py ...
```

Teacher/student checkpoint files are also excluded. If you want to reuse pretrained models, pass `--pretrained-core-root` and enable `--load-pretrained-students` or `--require-pretrained-teacher` as needed.

## Quick Start

Run a small smoke experiment first:

```bash
python src/run_zhang_paper_experiments.py ^
  --data-root "D:\path\to\zhang_dataset" ^
  --output-dir results/smoke ^
  --experiments main ^
  --methods no_kd fitnet ^
  --backbones gru ^
  --channels 0 ^
  --teacher-epochs 2 ^
  --student-epochs 2 ^
  --batch-size 32
```

Run the full paper-style comparison:

```bash
python src/run_zhang_paper_experiments.py ^
  --data-root "D:\path\to\zhang_dataset" ^
  --output-dir results/paper_baselines ^
  --experiments main ablation_student ablation_teacher analysis ^
  --methods fskd fitnet at sp vid rkd_d rkd_a rkd_full no_kd ^
  --backbones gru lstm ^
  --channels 0 1 2 3 ^
  --teacher-epochs 1000 ^
  --student-epochs 1000
```

Run the Top-50 reliable-sample FSKD variant:

```bash
python src/run_zhang_top50.py ^
  --data-root "D:\path\to\zhang_dataset" ^
  --output-dir results/top50 ^
  --experiments main ^
  --methods fskd ^
  --backbones gru lstm ^
  --channels 0 1 2 3 ^
  --fskd-reliable-select topk ^
  --fskd-topk-ratio 0.5
```

## PowerShell Scripts

The scripts in `scripts/` reproduce the larger experiment batches used during development. Before running them, edit the path variables near the top of each script, especially:

- `$DataRoot`: local dataset directory
- `$ResultRoot` / `$OutputRoot`: local output directory
- `$TeacherCache`: local teacher checkpoint cache, if used

Example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_paper_baselines_8tasks.ps1
```

## Outputs

Experiment runners write CSV/JSON outputs under the selected `--output-dir`, including per-run metrics, merged summaries, run configurations, and analysis artifacts. Generated outputs are ignored by Git by default.

## Notes for Reproducibility

- Default random seed is `1024`.
- Default split is 60% train, 20% test, 20% validation.
- Default full training uses 1000 teacher epochs and 1000 student epochs.
- `src/run_zhang_paper_experiments.py` contains the main baseline methods: NoKD, FitNet, AT, SP, VID, RKD-D, RKD-A, RKD-Full, and FSKD.
- `src/run_zhang_top50.py` contains the reliable-sample Top-k FSKD variant.

## Citation

If this code is used in a paper or report, please cite the corresponding project/paper when available.
