# Experiment Tracking Lab 2 (W&B + Keras)

This notebook trains a CNN on CIFAR-10 using TensorFlow/Keras and tracks experiments in Weights & Biases (W&B).

## Notebook

- Notebook file: `Lab2.ipynb`

## W&B URLs

- Project URL: `https://wandb.ai/abhinav241998-org/experiment-tracking-cifar10/workspace?nw=nwuserabhinav241998`

## Dataset Used

### CIFAR-10

- Source: `tensorflow.keras.datasets.cifar10`
- Classes (10): airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Image shape: `32 x 32 x 3`

### Effective dataset size in this notebook

The notebook reads `DATASET_SAMPLE_SIZE` from `.env.local` (default set to `20000`) and then computes:

- `n = min(sample, len(x_train), len(x_test))`

Because CIFAR-10 test split has `10000` images, the effective size becomes:

- Train: `10000`
- Test: `10000`

So even if `DATASET_SAMPLE_SIZE=20000`, current code uses `10000` per split.

## Model Architecture

The model is a custom CNN defined in `CIFAR10Trainer._build_model()`:

```text
+--------------------------------------+
| Input Block                           |
| Input Image: 32 x 32 x 3              |
+--------------------------------------+
                  |
                  v
+--------------------------------------+
| Augmentation Block                    |
| RandomFlip(horizontal)                |
| RandomRotation(0.05)                  |
+--------------------------------------+
                  |
                  v
+--------------------------------------+
| Feature Block 1                       |
| Conv2D(conv_1_filters, 3x3, relu)     |
| BatchNormalization                    |
| Conv2D(conv_1_filters, 3x3, relu)     |
| MaxPooling2D(2x2)                     |
| Dropout(dropout)                      |
+--------------------------------------+
                  |
                  v
+--------------------------------------+
| Feature Block 2                       |
| Conv2D(conv_2_filters, 3x3, relu)     |
| BatchNormalization                    |
| Conv2D(conv_2_filters, 3x3, relu)     |
| MaxPooling2D(2x2)                     |
| Dropout(dropout)                      |
+--------------------------------------+
                  |
                  v
+--------------------------------------+
| Classifier Block                      |
| GlobalAveragePooling2D                |
| Dense(dense_size, relu)               |
| Dropout(dropout)                      |
| Dense(10, softmax)                    |
+--------------------------------------+
                  |
                  v
+--------------------------------------+
| Output Block                          |
| Class probabilities for 10 classes    |
+--------------------------------------+
```

Layer flow summary:

1. Input layer: `(32, 32, 3)`
2. Data augmentation: `RandomFlip("horizontal")`, `RandomRotation(0.05)`
3. Feature extraction block 1: Conv -> BN -> Conv -> Pool -> Dropout
4. Feature extraction block 2: Conv -> BN -> Conv -> Pool -> Dropout
5. Classification head: GAP -> Dense -> Dropout -> Softmax(10)

### Optimizer and loss

- Optimizer: `Adam(learning_rate=OPT_LR)`
- Loss: `categorical_crossentropy`
- Metric: `accuracy`

## Environment-Driven Configuration

The notebook loads `.env.local` and supports these keys:

- W&B:
  - `WANDB_API_KEY`
  - `WANDB_PROJECT`
  - `WANDB_RUN_NAME`
  - `WANDB_MODE`
  - `WANDB_SILENT`
- TensorFlow:
  - `TF_CPP_MIN_LOG_LEVEL`
  - `TF_FORCE_GPU_ALLOW_GROWTH`
- Training:
  - `TRAIN_EPOCHS`
  - `TRAIN_BATCH_SIZE`
  - `DATASET_SAMPLE_SIZE`
  - `OPT_LR`
- Model:
  - `MODEL_DROPOUT`
  - `MODEL_CONV1_FILTERS`
  - `MODEL_CONV2_FILTERS`
  - `MODEL_DENSE_SIZE`

## What is logged to W&B

The notebook logs:

- Keras metrics each training step/epoch via `WandbMetricsLogger`
- Model checkpoints via `WandbModelCheckpoint`
- Learning rate at each epoch end (`LogLRCallback`)
- Sample predictions table with images (`LogSamplesCallback`)
- Confusion matrix each epoch (`ConfusionMatrixCallback`)
- Final test loss/accuracy
- Model artifact (`.keras`) and model summary text file

## How to run

1. Ensure dependencies are installed (`tensorflow`, `wandb`, `python-dotenv`).
2. Fill `.env.local` with your W&B key and desired config.
3. Open `Lab2.ipynb` and run cells from top to bottom.
