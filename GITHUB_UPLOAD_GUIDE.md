# GitHub Upload Guide

## Recommended repository details

- Repository name: `neuroocr-handwritten-digit-recognition`
- Visibility: Public
- Description: `Real-data handwritten digit recognition comparing classical ML with a NumPy CNN, confidence-aware inference, explainability, and a Streamlit demo.`
- Topics: `deep-learning`, `machine-learning`, `computer-vision`, `neural-network`, `cnn`, `numpy`, `scikit-learn`, `streamlit`, `ocr`, `python`

## Upload with the GitHub website

1. Extract the ZIP on Windows.
2. Open GitHub and create a new empty repository using the name above.
3. Do not add another README, `.gitignore`, or license during repository creation.
4. Select **uploading an existing file**.
5. Open the extracted project folder, select every file and folder inside it,
   and drag them into the GitHub upload area.
6. Wait until all files finish loading.
7. Use the commit message `Add complete NeuroOCR ML and deep learning project`.
8. Select **Commit changes**.

The repository root must directly show `README.md`, `app.py`, `run_all.py`,
`src`, `models`, `notebooks`, and `results`. Do not upload the outer ZIP as the
only repository file.

## Upload with Git commands

```powershell
git init
git add .
git commit -m "Add complete NeuroOCR ML and deep learning project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/neuroocr-handwritten-digit-recognition.git
git push -u origin main
```

Replace `YOUR-USERNAME` with the GitHub username before running the last two
commands.
