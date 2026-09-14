# Explainable Brain Tumor MRI Classification

An explainable four-class brain tumor MRI classification project built with PyTorch, transfer learning, CBAM attention, Grad-CAM, and Test-Time Augmentation.

## Project overview

I developed this project to compare multiple convolutional neural-network backbones for classifying brain MRI images as **glioma**, **meningioma**, **no tumor**, or **pituitary tumor**. The main model combines EfficientNet-B3 with a Convolutional Block Attention Module (CBAM), while Grad-CAM provides visual explanations of its predictions.

The project includes:

- A comparison of EfficientNet-B0, ResNet18, MobileNetV2, and EfficientNet-B3
- Channel and spatial attention through CBAM
- Two-stage transfer learning and backbone fine-tuning
- Label smoothing, Mixup, gradient clipping, and warmup-cosine scheduling
- Five-pass Test-Time Augmentation (TTA)
- Grad-CAM visualizations, ablation experiments, and error analysis
- An interactive single-image prediction workflow

## Recorded project results

These values were recorded during the completed term-project experiments on a separate test set of 1,600 images (400 per class). The notebook outputs have been cleared for a smaller, cleaner repository, so fresh runs may vary with the environment, hardware, and random operations.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| ResNet18 + CBAM | 88.00% | 88.04% |
| MobileNetV2 + CBAM | 91.25% | 91.22% |
| EfficientNet-B0 + CBAM | 91.56% | 91.54% |
| EfficientNet-B0 without CBAM | 91.75% | 91.72% |
| EfficientNet-B3 + CBAM | 92.44% | 92.42% |
| **EfficientNet-B3 + CBAM with TTA** | **94.06%** | **94.00%** |

The TTA result improved accuracy by 1.63 percentage points over standard EfficientNet-B3+CBAM inference. These are historical project results, not guaranteed results for every reproduction.

![Model accuracy comparison](results/model_comparison.png)

## Architecture

The primary pipeline is:

```text
MRI image
  -> augmentation and ImageNet normalization
  -> ImageNet-pretrained EfficientNet-B3 backbone
  -> CBAM channel attention
  -> CBAM spatial attention
  -> global average pooling
  -> regularized classification head
  -> four-class prediction
```

The backbone is initially frozen while the attention and classification layers learn. Its final blocks are then unfrozen for lower-learning-rate fine-tuning.

## Explainability

Grad-CAM highlights image regions that influence each prediction. The visualizations are useful for inspecting model behavior and identifying failure patterns, but they do not establish clinical validity or causal reasoning.

![Grad-CAM examples](results/gradcam_examples.png)

## Dataset

This project uses the [Brain Tumor MRI Dataset by Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). The dataset is not redistributed in this repository.

Download and extract the dataset yourself, then arrange it as follows:

```text
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The notebook uses `data/` by default. To keep the dataset elsewhere, set `BRAIN_TUMOR_DATASET_DIR` to the extracted dataset root.

```powershell
$env:BRAIN_TUMOR_DATASET_DIR = "D:\path\to\brain-tumor-mri-dataset"
```

## Requirements

- Python 3.10 or later
- PyTorch and torchvision
- A CUDA-capable GPU is strongly recommended for the complete experiment
- Enough storage for the dataset and locally generated model checkpoints

## Installation

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

Open [`notebooks/brain_tumor_pytorch_cbam.ipynb`](notebooks/brain_tumor_pytorch_cbam.ipynb) and run the cells in order. Set `QUICK_MODE = True` for a shorter functional run or leave it `False` for the complete experiment.

Generated checkpoints are written to `artifacts/models/` by default. Set `BRAIN_TUMOR_OUTPUT_DIR` to use another location.

## Repository structure

```text
.
├── data/
│   └── README.md
├── notebooks/
│   └── brain_tumor_pytorch_cbam.ipynb
├── results/
│   ├── efficientnet_b3_cbam_curves.png
│   ├── efficientnet_b3_cbam_tta_confusion.png
│   ├── gradcam_examples.png
│   ├── model_comparison.csv
│   └── model_comparison.png
├── .gitignore
├── README.md
└── requirements.txt
```

## Responsible use

This is an educational and research project. It has not been clinically validated and must not be used to diagnose a patient, guide treatment, or replace review by qualified medical professionals. Results obtained from this Kaggle dataset do not demonstrate generalization to real clinical environments.

## Maintainer

Shreyas Athinarapu

## License

No open-source license has been selected. Without a license, the source code remains under standard copyright protection and may not generally be reused, modified, or redistributed without permission.

