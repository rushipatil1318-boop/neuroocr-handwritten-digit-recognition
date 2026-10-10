# Generated results

| File | Meaning |
|---|---|
| `metrics.json` | Dataset, model, and confidence-policy metrics |
| `model_comparison.csv` | ML and CNN scores in table form |
| `test_predictions.csv` | Prediction and confidence for every test image |
| `training_curves.png` | CNN training and validation history |
| `confusion_matrix_ml.png` | Classical ML errors by digit class |
| `confusion_matrix_cnn.png` | CNN errors by digit class |
| `model_comparison.png` | Accuracy and macro-F1 comparison |
| `sample_predictions.png` | Low- and high-confidence examples |
| `saliency_examples.png` | Input-gradient explanations |
| `dataset_examples.png` | Real samples for all ten classes |
| `class_distribution.png` | Class balance |
| `run_environment.json` | Package versions from the completed run |

All results in this directory were produced by `python run_all.py` with random
seed 42. The executed notebook independently reproduces the same core result.
