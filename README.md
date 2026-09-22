# yelp-review-classifier
Yelp review classifier built while taking Stanford Pre-Collegiate Natural Language Processing course.


Predicting star ratings from review text with logistic regression and a neural network.

## Overview

This project scores a business review on a 1-5 star scale, directly from its text. This was a text classification task where, unlike typical categories, the labels are ordered (1 star is closer to 2 stars than to 5). The motivation was personal: I spend a lot of time reading reviews before deciding where to go, and wanted to see whether a model could read a review and predict roughly what rating it corresponds to.

## Dataset

- **Source:** HuggingFace's Yelp Review (Full) dataset
- **Size:** 650,000 labeled training reviews, 50,000 test reviews (label 0-4, representing 1-5 stars)
- **Sampling:** A subset was sampled from the full dataset for training/dev/test to keep iteration fast while experimenting

## Approach

Review text was converted to features with `CountVectorizer` (word counts), then fed into two different models to compare a simple approach against a more expressive one:

**Logistic Regression**
- Single linear layer (`scores = X·W + b`), softmax over 5 classes
- Fewer parameters → less prone to overfitting
- Can't capture word interactions or word order

**Neural Network**
- 2 hidden layers (ReLU) + softmax output layer
- Layer sizes tuned down specifically to reduce overfitting
- More expressive (captures word interactions), but more prone to overfitting on a
  small training set

**Loss function:** Cross-entropy — compares the predicted probability distribution to
the one-hot true label. Since the label is one-hot, only the predicted probability on
the correct class contributes directly to the loss, but softmax forces all predicted
probabilities to sum to 1, so pushing the correct class's probability up automatically
pushes the incorrect classes down.

## Results

| Model | Train Accuracy | Dev Accuracy | Test Accuracy |
|---|---|---|---|
| Logistic Regression | 53.75% | 50.1% | 52.25% |
| Neural Network | 64.79% | 50.5% | 53.05% |

The neural network shows a noticeably wider train/dev gap than logistic regression, a sign of memorizing training examples rather than generalizing. Hyperparameters explored included learning rate, epochs/iterations, `min_df` (vocabulary cutoff), and batch size; batch size mattered most for keeping training runnable while preserving accuracy.

## Challenges

- **Memory crashes & slow training:** converting word-count matrices to dense arrays
  (`.toarray()`) caused Colab to run out of RAM at larger sample sizes. Fixed by
  keeping the matrices sparse and shuffling once per epoch instead of random row
  indexing, then slicing consecutive batches.
- **Overfitting (neural network):** train accuracy climbed steadily while dev accuracy
  plateaued. Fixed by adding L2 regularization (weight decay), which discouraged
  individual weights from growing too large.

## Future Work

A natural next step would be reframing the task as 3-class classification (negative / neutral / positive) instead of 5-class — this is a substantially easier task and would likely raise accuracy meaningfully, while still being a useful signal.

## What I Learned

The overfitting behavior surprised me, as I didn't realize a large dataset could still be vulnerable to this issue. As I tried to increase the sample size to reverse this, I also came across the issue of my laptop being crashing - something that was never really a problem before. I learned how to problem solve while also learning how different NLP approaches work mathematically.

## Tech Stack

- Python, Google Colab
- scikit-learn (`CountVectorizer`, logistic regression)
- NumPy

## Acknowledgments

Built during Stanford Pre-Collegiate Studies' NLP course, with guidance from course instructor and AI-assisted coding tools.
