# MRGNN

MRGNN is an implementation for multivariate time-series anomaly detection. It combines graph-based modeling, Mamba-based temporal modeling, and Normalizing Flow.

## Environment Setup

A Linux environment with an NVIDIA GPU is recommended.

```bash
conda create -n mrgnn python=3.10 -y
conda activate mrgnn
```

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install causal-conv1d
pip install mamba-ssm --no-build-isolation
```

> The current model implementation uses CUDA directly, so an NVIDIA GPU is recommended.

## Data Preparation

Place the datasets under:

```text
Data/input/
```

Example directory structure:

```text
Data/input/
├── PSM/
├── WADI/
└── processed/      # MSL / SMAP / SMD
```

Some dataset paths are fixed in the current code, so it is recommended to keep the directory structure above.

## Training

The training entry point is:

```bash
python main.py [arguments]
```

Example for PSM:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --name PSM \
  --n_blocks 2 \
  --batch_size 256 \
  --window_size 60 \
  --train_split 0.6
```

You can also use the provided scripts:

```bash
bash runners/run_PSM.sh
bash runners/run_MSL.sh
bash runners/run_SMD.sh
bash runners/run_WADI.sh
```

Trained models are saved to:

```text
checkpoint/<dataset>/model.pth
```

## Testing

The testing entry point is:

```bash
python test.py [arguments]
```

Example for PSM:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --name PSM \
  --n_blocks 2 \
  --batch_size 256 \
  --window_size 60 \
  --train_split 0.6
```

You can also use the provided test scripts:

```bash
bash runners/run_PSM_test.sh
bash runners/run_MSL_test.sh
bash runners/run_SMD_test.sh
bash runners/run_WADI_test.sh
```

## Notes

Before training or testing, make sure that:

- CUDA and PyTorch are working correctly;
- `mamba_ssm` is installed successfully;
- datasets are placed in the corresponding `Data/input/` directories;
- the required checkpoint exists before running testing.
