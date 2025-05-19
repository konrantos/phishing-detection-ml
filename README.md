# Phishing Website Detection — Decision Trees & k-NN

> Course project for **Data Mining** (7th semester) — Department of Informatics and Telecommunications, University of Ioannina.

## Problem Description

This project focuses on classifying websites as **phishing** or **legitimate**, using machine learning algorithms. Two models are implemented:

- **Decision Trees**
- **k-Nearest Neighbors (k-NN)**

The classification is performed using the [PhishUCI dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset), which includes technical features of URLs such as structure, security status, redirects, and domain usage.

## Methodology

1. **Dataset Loading**
   - ARFF file loaded via `scipy.io.arff`
   - Converted to pandas DataFrame

2. **Data Preprocessing**
   - Feature conversion to integer
   - Feature/target split
   - Normalization for k-NN using `StandardScaler`

3. **Decision Tree**
   - Grid search over `max_leaf_nodes` in [200, 500]
   - 10-fold cross-validation to find best accuracy
   - Evaluation: Accuracy, Recall, F1-score
   - Confusion Matrix Visualization

4. **k-Nearest Neighbors**
   - Grid search over `k` from 1 to 15
   - 10-fold cross-validation for optimal `k`
   - Evaluation and confusion matrix like above


## How to Run

```bash
git clone https://github.com/konrantos/phishing-detection-ml.git
cd phishing-detection-ml
python phishing_detection.py
```

> Make sure the dataset file `Training Dataset.arff` is in the same directory.

## Dataset

- [PhishUCI: Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- Binary classification: `Result` column (`0 = phishing`, `1 = legitimate`)

## Tools & Libraries

- Python 3.11
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Scipy (for `.arff` format)

## License

MIT License

## Acknowledgements

- University of Ioannina — course project for *Data Mining*
