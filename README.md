# CIFAR-10 Image Classification with PyTorch

A small computer-vision project that trains a custom convolutional neural network (CNN) to classify CIFAR-10 images into ten classes.

## Project results

The original Google Colab run used a T4 GPU and PyTorch `2.11.0+cu130`. After 10 epochs, it reached **77.43% training accuracy** and **73.64% test accuracy** (test loss: **0.7770**).

| Dataset split | Images | Image size |
| --- | ---: | --- |
| Training | 50,000 | 3 × 32 × 32 RGB |
| Test | 10,000 | 3 × 32 × 32 RGB |

The model classified automobiles and ships relatively well. The confusion matrix shows frequent confusion between cats and dogs. The difference between the reported training and test accuracy is 3.79 percentage points.

> The numbers above are from the original Colab run. Training again can produce different results because of random initialization and runtime differences.

## Model

The network has three convolution and max-pooling blocks, followed by two fully connected layers. It uses ReLU activations and dropout (`p=0.4`). The input is converted to a tensor with `ToTensor()`. Training uses cross-entropy loss, Adam with a learning rate of `0.001`, and batches of 128 images.

```text
3-channel 32×32 image
  → Conv2d(3, 32) → ReLU → MaxPool2d
  → Conv2d(32, 64) → ReLU → MaxPool2d
  → Conv2d(64, 128) → ReLU → MaxPool2d
  → Flatten → Linear(2048, 256) → ReLU → Dropout(0.4)
  → Linear(256, 10)
```

## Results from the original run

### Training history

![Training loss and accuracy over 10 epochs](assets/training_history.png)

### Test-set confusion matrix

![CIFAR-10 test-set confusion matrix](assets/confusion_matrix.png)

Rows in the confusion matrix are actual classes; columns are predicted classes. The diagonal represents correct predictions.

## Run the project

1. Clone or download this repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train and evaluate the model:

   ```bash
   python train.py --epochs 10 --batch-size 128 --learning-rate 0.001
   ```

The CIFAR-10 dataset downloads automatically the first time you run the script. Use a CUDA-enabled PyTorch installation to train on a supported GPU; otherwise, the script falls back to CPU.

## Generated files

The script writes these files to `outputs/`:

- `cifar10_cnn.pth` — trained model weights
- `training_history.png` — loss and training accuracy by epoch
- `confusion_matrix.png` — test-set predictions by class

The original checkpoint from Colab was not available in the files used to prepare this repository. Run the script to generate a checkpoint, or add the original `cifar10_cnn.pth` from Colab if you want to publish those exact weights.

## Repository contents

- `CIFAR10_Image_Classification.ipynb` — Colab-friendly notebook to run the project
- `train.py` — training and evaluation script
- `assets/` — result figures from the original run
- `requirements.txt` — Python dependencies

## Technologies

Python · PyTorch · Torchvision · Matplotlib · CIFAR-10 · Google Colab
