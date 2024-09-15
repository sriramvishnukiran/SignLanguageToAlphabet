# Sign Language to Alphabet converter

This project utilizes Convolutional Neural Networks (CNN) and custom data creation to interpret sign language in real-time.

## Table of Contents

- [Sign Language to Alphabet converter](#sign-language-to-alphabet-converter)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Creating the dataset](#creating-the-dataset)
    - [Training the model](#training-the-model)
    - [Using live video feed](#using-live-video-feed)
  - [Libraries Used](#libraries-used)
  - [References](#references)

## Introduction

This project is part of the CS335 (AI/ML Lab) course instructed by Prof. Preethi Jyothi.

Collaboratively developed by:

- Aditya Raj
- Hruday Nandan Tudu
- Modulla Hrushikesh Reddy
- Shikhar Parmar
- Duggineni Venkata Paneesh

## Installation

To get started with the application, clone the repository and install the necessary dependencies using the `requirements.txt` file.

```bash
# Example installation steps
git clone https://github.com/A-raj468/SignLangToAlphabet.git
cd SignLangToAlphabet
pip install -r requirements.txt
```

## Usage

### Dataset

You can begin by downloading the [dataset](https://iitbacin-my.sharepoint.com/:f:/g/personal/210050005_iitb_ac_in/Evck4s6jkDpAuCZZm_heMEkBUh4TP8v8o_jw1ne6FMt0bQ?e=8vN32T).

Or

By creating your own dataset using `collect_images.py`.

```bash
python collect_images.py
```

To begin creating dataset press 'S', then press 'C' to capture an image.

To extract the joints from hand images for FNN model use `create_dataset.py`

```bash
python create_dataset.py
```

### Training the model

For training the CNN model use `CNN_model.py`,

```bash
python CNN_model.py
```

For training the FNN model use `FNN_model.py`,

```bash
python FNN_model.py
```

### Using live video feed

For prediction using CNN, run the `inference_classifier_CNN.py` to start the live feed and show sign language with one hand.

```bash
python inference_classifier_CNN.py
```

For prediction using FNN, run the `inference_classifier_FNN.py` to start the live feed and show sign language with one hand.

```bash
python inference_classifier_FNN.py
```

Quit by presing 'Q'.

## Libraries Used

- PyTorch
- Tqdm

## References

- [Make dataset](https://github.com/computervisioneng/sign-language-detector-python)
