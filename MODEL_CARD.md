# NeuroOCR Model Card

## Intended use

NeuroOCR is an educational prototype for recognizing a single centered
handwritten digit. A realistic use case is assisting human data entry from
simple forms, meter readings, labels, or receipts.

## Model

The main model is a compact convolutional neural network implemented directly
with NumPy. It contains one 3 × 3 convolution, ReLU activation, 2 × 2 max
pooling, a 64-unit dense layer, and a 10-class softmax output. Adam optimization,
L2 regularization, validation monitoring, and best-weight restoration are
implemented in the project code.

## Evaluation

The held-out test set contains 270 images and is never used for fitting or
model selection. The saved run achieved 99.63% test accuracy and 0.9962 macro
F1. Results are deterministic for seed 42 on the recorded environment.

## Confidence policy

A prediction is accepted only when its maximum softmax probability is at least
0.85. On the saved test run this retained 96.30% of images and achieved 99.62%
accuracy on accepted predictions. Low-confidence inputs are routed to manual
review instead of being silently treated as reliable.

## Limitations

- Training images are only 8 × 8 pixels and are already centered.
- The model recognizes one digit, not complete words or multi-digit documents.
- Phone photos, unusual writing, clutter, rotation, and poor lighting can reduce
  performance.
- Test accuracy on this compact benchmark does not prove production readiness.
- The model must not be used for identity, financial, medical, legal, or other
  high-stakes automated decisions.
