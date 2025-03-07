x# Naive Bayes Classifier for Iris Dataset

## Overview
This project implements two Naive Bayes classifiers:
- **Discrete Naive Bayes** (with feature discretization)
- **Gaussian Naive Bayes** (using normal distribution for likelihood estimation)

The classifiers are tested on the Iris dataset, and their accuracy is evaluated with different training set sizes and random states.

## Features
- **Custom Naive Bayes Implementation**: Implements both discrete and Gaussian Naive Bayes classifiers from scratch.
- **Data Discretization**: Converts continuous features into discrete bins for the discrete classifier.
- **Custom Likelihood Calculation**: Computes prior and likelihood probabilities for classification.
- **Accuracy Comparison**: Compares both classifiers over varying training set sizes.
- **Visualization**: Plots accuracy vs. training set size.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/naive-bayes-iris.git
   cd naive-bayes-iris
   ```
2. Install dependencies:
   ```sh
   pip install numpy matplotlib scikit-learn
   ```
3. Run the script:
   ```sh
   python naive_bayes.py
   ```

## Usage
### Define Classifiers
- **NaiveBayes**: Uses discrete bins for feature representation.
- **GaussianNaiveBayes**: Uses a Gaussian distribution for likelihood estimation.

```python
nb_classifier = NaiveBayes()
nb_classifier.build_classifier(x_train, y_train)

gnb_classifier = GaussianNaiveBayes()
gnb_classifier.build_classifier(x_train, y_train)
```

### Evaluate Accuracy
```python
nb_accuracy, gnb_accuracy = calculate_accuracy(x_train, x_test, y_train, y_test)
print("Naive Bayes Accuracy:", nb_accuracy)
print("Gaussian Naive Bayes Accuracy:", gnb_accuracy)
```

### Plot Accuracy vs Train Size
```python
test_accuracy_for_different_sizes_and_random_state(x, y, random_states=[123, 42, 7, 99])
```

## Parameters
- `bins`: Number of bins for discretization in **NaiveBayes**.
- `random_state`: Seed for reproducibility in train-test splitting.
- `train_size`: Proportion of data used for training.
- `test_size`: Proportion of data used for testing.

## Output
- **Accuracy Scores**: Displays the classification accuracy of both models.
- **Plots**: Generates accuracy vs. train size plots for different random seeds.
---
Made by Anna Wierzbik

