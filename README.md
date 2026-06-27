# Zhang Gas Sensor Teacher-Student Baseline

Minimal code for gas sensor concentration regression on Zhang's dataset. This repository only keeps the basic dataset loader, teacher model, student model, and a simple teacher-student distillation training script.

> Dataset files, checkpoints, and generated results are not included.

## Structure

```text
.
├── src/
│   ├── data.py                    # Dataset loading and preprocessing
│   ├── model.py                   # Teacher/student network definitions
│   └── train_teacher_student.py   # Basic teacher-student training script
├── results/                       # Local output folder placeholder
├── requirements.txt
└── README.md
```

## Environment

Recommended:

- Python 3.9+
- PyTorch
- CUDA GPU is optional but recommended for training

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you need a specific CUDA version of PyTorch, install PyTorch first from the official PyTorch website, then run `pip install -r requirements.txt`.

## Dataset

Pass the dataset folder with `--data-root`. The expected layout is one folder per concentration label, where the folder name contains three comma-separated target values:

```text
/path/to/zhang_dataset/
├── 100,20,50/
│   ├── sample1.txt
│   └── sample2.txt
├── 200,40,100/
│   └── sample3.txt
└── ...
```

Each `.txt`/`.TXT` file should contain sensor response data. The loader uses the first 4 sensor columns and resizes each sample to `--sequence-length`.

## Run

Example training command:

```bash
python src/train_teacher_student.py ^
  --data-root "D:\path\to\zhang_dataset" ^
  --output-dir results/basic ^
  --teacher-epochs 100 ^
  --student-epochs 100 ^
  --batch-size 32
```

For a quick test, use fewer epochs:

```bash
python src/train_teacher_student.py ^
  --data-root "D:\path\to\zhang_dataset" ^
  --output-dir results/smoke ^
  --teacher-epochs 2 ^
  --student-epochs 2
```

## What the Script Does

1. Loads Zhang gas sensor samples with `ZhangGasDataset`.
2. Splits data into 60% train, 20% validation, and 20% test.
3. Trains a larger teacher model with supervised MSE loss.
4. Trains a smaller student model using supervised loss plus teacher prediction distillation loss.
5. Saves outputs to `--output-dir`:
   - `teacher.pt`
   - `student.pt`
   - `metrics.json`

## Main Arguments

- `--data-root`: dataset directory, required.
- `--output-dir`: output directory, default `results/basic`.
- `--sequence-length`: sequence length, default `200`.
- `--teacher-epochs`: teacher training epochs, default `100`.
- `--student-epochs`: student training epochs, default `100`.
- `--teacher-dim`: teacher hidden dimension, default `64`.
- `--student-dim`: student hidden dimension, default `32`.
- `--rnn-type`: `gru` or `lstm`, default `gru`.
- `--distill-alpha`: weight of teacher prediction loss, default `0.5`.

## Notes

This is a clean baseline version intended for GitHub release. It does not include comparison experiments, ablation experiments, grid search scripts, paper table generation, original datasets, or trained checkpoints.
