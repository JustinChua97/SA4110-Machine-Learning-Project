# SA4110 Machine Learning Project — Fruit Image Classifier

A Convolutional Neural Network (CNN) that classifies images of fruit into three
classes: **apple**, **banana**, and **orange**. Built for the GDIPSA 26 SA4110
Machine Learning module as a team project.

---

## Project context

The goal of this project is to build an image classifier that can correctly
identify a fruit from a photo. We were given a small labelled dataset and asked
to train a CNN, then experiment with ways to improve its accuracy and document
what worked.

| | |
|---|---|
| **Task** | Multi-class image classification (3 classes) |
| **Classes** | Apple, Banana, Orange |
| **Dataset** | `Train.zip` (~220 images) and `Test.zip` (~55 images) |
| **Model type** | Convolutional Neural Network (CNN) |
| **Final test accuracy** | **>90%** |

The dataset is deliberately small, which makes overfitting the central challenge
of the project. Most of the design work was about getting a model to generalise
well despite having very little data to learn from.

---

## Tech stack

- **Python**
- **TensorFlow / Keras** — model building and training
- **scikit-learn** — confusion matrix and classification report
- **Matplotlib / Seaborn** — plots and visualisation
- **Pillow (PIL) / NumPy** — image loading and array handling

---

## Repository structure

```
SA4110-Machine-Learning-Project/
├── fruit_classifier.ipynb   # Main project notebook (the 3 fruit models)
├── mnist_keras.ipynb        # Reference notebook (learning CNNs on MNIST)
├── train.zip                # Raw training images
├── test.zip                 # Raw test images
├── train/train/             # Extracted training images
├── test/test/               # Extracted test images
└── dataset/                 # Images sorted into class folders (apple/banana/orange)
```

---

## Why MNIST first? (reference notebook)

Before tackling the fruit dataset, we built a CNN on the classic **MNIST**
handwritten-digit dataset in `mnist_keras.ipynb`. MNIST is the standard
"hello world" of computer vision: it is clean, well-labelled, and large, so it
let us learn the full Keras workflow — loading data, normalising pixels,
building Conv/MaxPool/Dense layers, compiling, training, and evaluating — without
the complications of a small, messy dataset.

The MNIST model uses a simple architecture:

```
Input(28x28x1) → Conv(16) → MaxPool → Conv(32) → MaxPool → Flatten → Dense(128) → Dense(10, softmax)
```

It reaches roughly 99% test accuracy on MNIST. This served as our **template**:
the fruit classifier reuses the same core CNN building blocks, adapted from
grayscale digits (28×28×1, 10 classes) to colour fruit images (RGB, 3 classes).

---

## Design process

The workflow in `fruit_classifier.ipynb` follows these stages:

1. **Data preparation** — Unzip `train.zip` / `test.zip`, then sort every image
   into a class folder (`dataset/train/apple`, `.../banana`, `.../orange`) based
   on its filename prefix, so Keras `flow_from_directory` can read the labels
   automatically.
2. **Exploratory analysis** — Plot the class distribution (training set is
   roughly balanced: apple 75, banana 73, orange 72) and view sample images.
3. **Label check** — Manually inspect images that look ambiguous to confirm they
   are labelled correctly.
4. **Iterative modelling** — Train three increasingly capable CNNs, evaluating
   each on the held-out test set before deciding what to change next.
5. **Comparison and evaluation** — Compare all three models, then produce a
   confusion matrix and classification report for the best one.

---

## The three models

Each model was a deliberate response to a weakness observed in the previous one.

### Model 1 — Simple CNN (baseline)

A minimal network to establish a baseline.

```
Input(32x32x3) → Conv(8) → MaxPool → Conv(16) → MaxPool → Flatten → Dense(32) → Dense(3, softmax)
```

- Image size: **32×32**, no augmentation, 2 conv layers, 10 epochs.
- **Test accuracy: ~80%.**
- *Takeaway:* Workable, but 32×32 is small and loses a lot of detail.

### Model 2 — Deeper CNN

Added depth and regularisation to learn richer features.

```
Input(32x32x3) → Conv(16) → MaxPool → Conv(32) → MaxPool → Conv(64) → MaxPool
              → Flatten → Dense(64) → Dropout(0.3) → Dense(3, softmax)
```

- Image size: **32×32**, no augmentation, 3 conv layers + dropout, 10 epochs.
- **Test accuracy: ~85%.**
- *Takeaway:* A small improvement, but training accuracy ran far ahead of
  validation accuracy — clear **overfitting** caused by the tiny dataset.

### Model 3 — Deeper CNN + Image Augmentation (final model)

To fix overfitting, we generated more variety from the existing images using
**data augmentation** (random rotation, horizontal flip, width/height shift,
zoom, and brightness changes) and increased the input resolution.

```
Input(64x64x3) → Conv(32) → MaxPool → Conv(64) → MaxPool → Conv(128) → MaxPool
              → Flatten → Dense(128) → Dropout(0.5) → Dense(3, softmax)
```

- Image size: **64×64**, full augmentation, 3 conv layers + dropout, 30 epochs.
- **Test accuracy: >90%** (best result).
- *Takeaway:* Augmentation gave the biggest single boost — the model effectively
  sees "new" images every epoch — and the train/validation curves moved much
  closer together, showing reduced overfitting. The larger 64×64 input and
  stronger dropout also helped.

---

## Results summary

| Model | Description | Image size | Augmentation | Test accuracy |
|-------|-------------|:----------:|:------------:|:-------------:|
| Model 1 | Simple CNN (2 conv layers) | 32×32 | No | ~80% |
| Model 2 | Deeper CNN (3 conv + dropout) | 32×32 | No | ~85% |
| **Model 3** | **Deeper CNN (final)** | **64×64** | **Yes** | **>90%** ✅ |

Random guessing on 3 balanced classes would score ~33%, so all three models
learn meaningful features; the improvements come from depth, resolution, and —
most importantly — augmentation.

---

## Key learnings

- **Augmentation is the biggest lever** on a small dataset. It is the closest
  thing to "more data" without collecting any.
- **Image resolution matters** — moving from 32×32 to 64×64 let the model see
  more fruit detail.
- **Depth alone overfits** on a small dataset; it needs dropout and augmentation
  to actually pay off.
- **Dropout** meaningfully reduced the gap between training and validation
  accuracy.

---

## How to run

1. Clone the repository:
   ```bash
   git clone https://github.com/JustinChua97/SA4110-Machine-Learning-Project.git
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow scikit-learn matplotlib seaborn pillow numpy
   ```
3. Open `fruit_classifier.ipynb` in Jupyter or VS Code and run the cells in
   order. The notebook unzips the data and builds the `dataset/` folders
   automatically.

> Run `mnist_keras.ipynb` first if you want to follow the same learning path we
> took — MNIST → fruit classifier.

---

*GDIPSA 26 · SA4110 Machine Learning · Team Project*
