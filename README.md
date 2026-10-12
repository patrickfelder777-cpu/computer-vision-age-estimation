# Computer Vision Age Estimation

**Patrick Felder | TripleTen coursework**

Implemented exploratory data analysis and a ResNet50 transfer-learning model for age regression. The project demonstrates model architecture, batched image loading, data augmentation, and interpretation of training diagnostics.

## What I implemented

- Explored 7,591 labeled facial images, ages 1–100; checked missing values, duplicates, and the age distribution.
- Built image generators with 224 × 224 inputs, a 25% validation subset, batch size 32, rescaling, and horizontal-flip augmentation for training.
- Defined an ImageNet-pretrained ResNet50 backbone without its classification head, followed by global average pooling, dropout 0.2, and a one-unit ReLU regression head.
- Compiled with Adam at learning rate 0.0001, MSE loss, and MAE reporting; provided functions for 20-epoch training.

## Result provenance

**The supplied training log is course-provided reference output, not a verified personal run.** The original review explicitly describes it as reference output from course materials, and its source has been confirmed. This repository does not claim that my implementation achieved these scores.

| Reference statistic | MAE in years | Epoch |
|---|---:|---:|
| Best validation | 6.6419 | 17 |
| Final validation | 7.6512 | 20 |
| Final training | 3.1785 | 20 |

There is no separate held-out test score. Despite the function name `load_test`, it returns the validation subset. The reference log's 356 steps per epoch differ from the approximately 178 implied by this code's dataset split and batch size, so it should not be treated as a reproduced execution.

## Interpretation

The reference results illustrate a training/validation gap and a higher final validation error than the best epoch. Early stopping and saving the best checkpoint are future improvements, not capabilities implemented in the original functions. MAE alone does not establish reliable performance around an age threshold or across demographic groups. No retail deployment, automated eligibility decision, or measured business impact is claimed.

## Run with your own data

Use a Python virtual environment with TensorFlow and the dependencies in `requirements.txt`. The original environment versions were not captured; dependencies are not pinned, and compatibility may require adjustments. The code retains the coursework `ImageDataGenerator` API.

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Place an authorized dataset in this structure:

```text
data/faces/labels.csv
data/faces/final_files/<image filenames>
```

`labels.csv` must contain `file_name` and `real_age`. Set `FACE_DATA_DIR` to another location for the notebook. The original `/datasets/faces` path is supported as a fallback. The image dataset is not redistributed, and example face-image outputs have been removed from the portfolio notebook.

To run a new training job:

```bash
python train_model.py --data-dir data/faces --epochs 20 --output models/age_estimation.keras
```

The script downloads ImageNet weights if not cached. A GPU is useful for training; hardware setup depends on the installed TensorFlow environment. This command saves the final model; it does not save the best epoch separately.

## Files

- `age_estimation.ipynb`: edited portfolio notebook with provenance labels.
- `train_model.py`: portable model functions and a training entry point.
- `reference_training_log.txt`: explicitly labeled course log.
- `reference_metrics.csv`: values parsed from that reference log.
- `requirements.txt`: dependencies.

## Verification

All code cells and the training script passed Python syntax checks. Twenty reference epochs were parsed and checked against the recorded best and final MAE. Training was not run because the image dataset and GPU execution environment were not supplied.
