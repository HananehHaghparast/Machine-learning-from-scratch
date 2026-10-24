# ML Core — From Linear Models to Neural Networks

### An essay-style README for a from-scratch machine learning notebook

This repository is an educational machine learning project that traces a deliberate path from simple linear models to neural networks. Rather than treating machine learning as a collection of library calls, the accompanying notebook, `ml_core.ipynb`, asks a more fundamental question: **what do these models actually do, and where do they break?**

The central narrative is simple but powerful. Linear regression gives us a clean introduction to gradient descent and loss minimization. The perceptron shows how a linear decision boundary can separate classes. Logistic regression introduces probabilities and a more principled classification objective. Polynomial feature engineering demonstrates how a linear model can be pushed toward nonlinear behavior. Finally, neural networks reveal how hidden layers and nonlinear activations allow a model to learn representations rather than relying on hand-crafted transformations.

Throughout the notebook, the emphasis is on **implementation from scratch using NumPy**. No pre-built estimators from `scikit-learn` are used for learning. Utility tools such as `PolynomialFeatures` and dataset generators like `make_regression`, `make_blobs`, and `make_circles` are used only for data preparation. This keeps the focus on the internal mechanics: forward passes, gradients, updates, decision boundaries, and failure modes.

---

## Why This Project Exists

Many introductory machine learning workflows begin with `model.fit()` and end with a score. That is useful, but it can hide the ideas that make machine learning interesting. This notebook takes the opposite approach. It rebuilds the core models by hand so that the reader can see:

- how predictions are formed,
- how loss functions shape learning,
- how gradients are derived and applied,
- how model assumptions fail on real or complex data,
- and why neural networks are not magic, but a natural extension of the same ideas.

The project is therefore both a tutorial and a laboratory. Each section introduces a model, applies it to a carefully chosen dataset, and examines both its successes and its limitations.

---

## What the Notebook Covers

### 1. Linear Regression

The notebook begins with linear regression for continuous targets. It implements gradient descent for mean squared error, includes feature standardization for stable training, and maps learned parameters back to the original feature space. The model is then applied to a synthetic regression dataset.

The section also demonstrates two important lessons:

- **Feature engineering works.** A quadratic relationship cannot be captured by a straight line, but polynomial features allow the same linear model to fit a curve.
- **Linear regression is not a classifier.** When applied to a binary classification problem with outliers, the squared-error objective is dominated by extreme points. The resulting decision threshold shifts, and otherwise reasonable points become misclassified.

This failure is not a bug. It is a consequence of optimizing the wrong objective for the task.

### 2. Perceptron

The perceptron is introduced as the simplest linear classifier. The implementation uses online, sample-by-sample updates, shuffles the data each epoch, and updates weights only when a point is misclassified. On linearly separable data, it converges to a perfect boundary. On overlapping or nonlinearly separable data, it does not.

This section makes an important conceptual point: **a linear classifier can only succeed when the data admits a linear separation.** When that assumption fails, no amount of perceptron training will fix it.

### 3. Logistic Regression

Logistic regression replaces hard 0/1 outputs with probabilities. The notebook implements the sigmoid function with numerical clipping, binary cross-entropy, and gradient descent. It also provides `predict_proba` and a `score` method so that the model can be evaluated and visualized consistently with the other models.

The comparison with the perceptron is direct:

- the perceptron reacts only to mistakes,
- logistic regression updates on every sample, scaled by the error,
- and logistic regression produces a probabilistic boundary rather than an arbitrary linear separator.

### 4. Neural Networks

The final destination is the neural network. Hidden layers and nonlinear activation functions allow the model to learn complex decision boundaries that cannot be expressed by a single linear separator. The notebook discusses forward propagation, backpropagation, and the role of nonlinearity. It places neural networks in context: they are not disconnected from the earlier models. They are what happens when we stack differentiable linear transformations and nonlinear activations, then train the whole system with gradient descent.

The datasets used in the notebook — especially the spiral and checkerboard datasets — are chosen to make this progression visible. A linear model fails on these problems. A neural network, given enough capacity, can begin to succeed.

---

## Datasets Used

The notebook uses a combination of synthetic and real-world data:

- **Linear regression data** from `make_regression`
- **Nonlinear regression data** generated as a quadratic curve with noise
- **1D classification data with outliers**
- **Linearly separable blobs** from `make_blobs`
- **Overlapping blobs** to show the limits of hard-margin linear classification
- **Circles** from `make_circles`
- **A four-class spiral dataset**
- **A checkerboard dataset**
- **MNIST handwritten digits**, loaded locally from the `archive` directory

Each dataset is not just a test case. It is an argument. The linear regression dataset establishes the baseline. The outlier dataset shows why squared error is inappropriate for classification. The overlapping blobs show why the perceptron cannot always converge. The spiral and checkerboard datasets show why representation learning matters.

---

## Design Principles

Several principles guide the notebook:

1. **Transparency over convenience.** Models are implemented with NumPy so that every update rule is visible.
2. **Failure as a teaching tool.** The notebook does not hide model limitations. It deliberately shows where linear regression, perceptron, and logistic regression struggle.
3. **Consistent model interfaces.** The implementations follow familiar conventions such as `fit`, `predict`, `predict_proba`, `score`, `coef_`, and `intercept_`.
4. **Reproducibility.** Random seeds are used for synthetic datasets and model initialization.
5. **Conceptual progression.** Each model is introduced because it solves a limitation of the previous one, not because it appears in a syllabus.

---

## Key Takeaways

This project is built around a few enduring ideas in machine learning:

- **A model is only as good as its inductive bias.** Linear models assume linearity. When that assumption is false, they fail in predictable ways.
- **Loss functions encode priorities.** Mean squared error is sensitive to outliers; cross-entropy is better aligned with classification.
- **Feature engineering is powerful but manual.** Polynomial features can help, but they require human insight and can become unwieldy.
- **Neural networks learn features.** Hidden layers allow the model to discover nonlinear representations instead of relying entirely on hand-designed transformations.
- **Understanding failure is part of understanding the model.** The most instructive moments in the notebook are often the ones where a model breaks.

---


---

## Who This Is For

This notebook is intended for students, self-learners, and practitioners who want to move beyond surface-level familiarity with machine learning. It is especially useful for readers who have used `scikit-learn` before but want to understand what happens underneath the API. It is not a production library. It is a learning artifact: deliberate, explicit, and occasionally deliberately naive, so that the mechanics remain visible.

---


