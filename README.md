# Brain Tumor Classification with CNN

A TensorFlow/Keras notebook project for classifying brain MRI images into four categories: glioma, meningioma, no tumor, and pituitary tumor. The notebook trains a convolutional neural network, evaluates it on a held-out test set, plots training metrics, and predicts the class of `sample.jpg`.

## Project files

- `Untitled1.ipynb` — model training, evaluation, and single-image inference workflow.
- `archive/Training/` — training images organized into class folders. The notebook reserves 20% of this data for validation.
- `archive/Testing/` — test images organized into the same class folders.
- `brain_tumor_cnn.keras` — saved Keras model (created/overwritten when the training cell runs).
- `sample.jpg` — example image used for inference.

Expected dataset layout:

```text
archive/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

## Requirements

Python 3.9+ and the packages used in the notebook:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn jupyter
```

TensorFlow installation requirements vary by operating system and hardware. Use the [official TensorFlow installation guide](https://www.tensorflow.org/install) if the command above does not work for your setup.

## Run the notebook

1. Clone or download this project and make sure the dataset folders match the layout above.
2. Install the requirements.
3. Start Jupyter and open `Untitled1.ipynb`:

   ```bash
   jupyter notebook
   ```

4. Before running the training cell, define the dataset paths and image settings. For example, when running from the project root:

   ```python
   train_path = "archive/Training"
   test_path = "archive/Testing"
   IMG_SIZE = (224, 224)
   BATCH_SIZE = 32
   ```

5. Run the cells in order. Training uses 30 epochs and saves the model as `brain_tumor_cnn.keras` in the current working directory.

The notebook uses images resized to 224 × 224 pixels and scales pixel values to [0, 1]. Training augmentation includes rotation, zoom, shear, and horizontal flips. Its CNN has four convolutional blocks, batch normalization, max pooling, dropout, and a four-class softmax output.

## Predict with the example image

The notebook's inference cell loads `sample.jpg` from the current working directory, resizes it to 224 × 224, and prints the predicted class and softmax score. To use another image, update `img_path` to its path. The model must already be trained in the current notebook session, or loaded from disk first:

```python
from tensorflow.keras.models import load_model
model = load_model("brain_tumor_cnn.keras")
```

The inference class order in the notebook is `Glioma`, `Meningioma`, `No Tumor`, `Pituitary`.

## Evaluation

The notebook evaluates the model on `archive/Testing/`, plots training and validation accuracy/loss, and prints a confusion matrix and classification report. Reported performance depends on the dataset and training run; no accuracy claim is made here.

## Important

This is an educational image-classification project, not a medical diagnostic tool. Do not use its predictions for clinical decisions. MRI scans should be interpreted by qualified medical professionals.
