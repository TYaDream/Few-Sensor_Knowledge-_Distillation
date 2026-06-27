# Zhang Gas Sensor Teacher-Student Baseline

This repository provides a minimal teacher-student distillation baseline for gas concentration regression on Zhang's gas sensor dataset. It only keeps the essential code: dataset loading, teacher/student model definitions, and a basic training script.

This repository does not include the dataset, generated results, trained weights, ablation experiments, or comparison experiments.

## Project Structure

```text
.
├── src/
│   ├── data.py                    # Dataset loading and preprocessing
│   ├── model.py                   # Teacher and student model definitions
│   └── train_teacher_student.py   # Basic teacher-student training script
├── results/                       # Local output folder placeholder
├── requirements.txt               # Python dependencies
└── README.md
```

## Environment

Python 3.9 or later is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you want to train with a GPU, install the PyTorch version that matches your CUDA environment first, then install the remaining dependencies.

## Dataset Format

Use `--data-root` to specify the dataset directory. The dataset should be organized by concentration labels. Each subfolder name should contain three comma-separated target values, for example:

```text
/path/to/zhang_dataset/
├── 100,20,50/
│   ├── sample1.txt
│   └── sample2.txt
├── 200,40,100/
│   └── sample3.txt
└── ...
```

Each `.txt` or `.TXT` file contains sensor response data. The current loader uses the first 4 sensor columns and resizes each sample to `--sequence-length`.

## Usage

Basic training command:

```bash
python src/train_teacher_student.py ^
  --data-root "D:\path\to\zhang_dataset" ^
  --output-dir results/basic ^
  --teacher-epochs 100 ^
  --student-epochs 100 ^
  --batch-size 32
```

Quick smoke test:

```bash
python src/train_teacher_student.py ^
  --data-root "D:\path\to\zhang_dataset" ^
  --output-dir results/smoke ^
  --teacher-epochs 2 ^
  --student-epochs 2
```

## Training Pipeline

`src/train_teacher_student.py` performs the following steps:

1. Loads samples with `ZhangGasDataset`.
2. Splits the dataset into 60% training, 20% validation, and 20% testing.
3. Trains the teacher model with supervised MSE loss.
4. Trains the student model after the teacher is trained.
5. Uses both supervised label loss and teacher prediction distillation loss for student training.
6. Saves model weights and test metrics to the output directory.

The output directory contains:

```text
results/basic/
├── teacher.pt
├── student.pt
└── metrics.json
```

## Main Arguments

- `--data-root`: Path to the dataset directory. Required.
- `--output-dir`: Output directory. Default: `results/basic`.
- `--sequence-length`: Input sequence length. Default: `200`.
- `--batch-size`: Batch size. Default: `32`.
- `--teacher-epochs`: Number of teacher training epochs. Default: `100`.
- `--student-epochs`: Number of student training epochs. Default: `100`.
- `--teacher-dim`: Hidden dimension of the teacher model. Default: `64`.
- `--student-dim`: Hidden dimension of the student model. Default: `32`.
- `--rnn-type`: Recurrent backbone type, either `gru` or `lstm`. Default: `gru`.
- `--distill-alpha`: Weight of the teacher prediction distillation loss. Default: `0.5`.

## Notes

This is a clean minimal release. It does not include:

- Ablation experiments
- Comparison experiments
- Top-k / Top-50 experiments
- Hyperparameter grid search scripts
- Paper table generation scripts
- Original dataset files
- Trained model checkpoints

To reproduce results, prepare the dataset locally and adjust the number of epochs and model parameters as needed.
