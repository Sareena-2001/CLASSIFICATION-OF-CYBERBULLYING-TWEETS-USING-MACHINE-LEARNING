# Classification of Cyberbullying Tweets Using Machine Learning

This project builds a traditional machine-learning pipeline to classify short text comments as cyberbullying or non-cyberbullying. It combines text preprocessing, TF-IDF feature extraction, several scikit-learn classifiers, and evaluation metrics with visualizations.

## Project Overview

The code compares seven classifiers:

- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Multinomial Naive Bayes
- Logistic Regression

The project includes two versions of the analysis: one that compares classifier metrics and another that also plots a confusion matrix for each classifier.

## Dataset

The supplied `CyberBullying Comments Dataset.csv` contains **11,100 rows** and two columns.

| Column | Description |
|---|---|
| `Text` | Text of a comment |
| `CB_Label` | Binary class label: `0` or `1` |

The supplied CSV has no missing values. The labels are balanced:

| Label | Meaning in the project code | Records |
|---|---|---:|
| `0` | Non-cyberbullying | 5,550 |
| `1` | Cyberbullying | 5,550 |

## Workflow

1. Load the CSV dataset with pandas.
2. Clean the text by converting it to lowercase, removing URLs and non-letter characters, and tokenizing it.
3. Remove English stop words.
4. Apply Porter stemming and WordNet lemmatization.
5. Split the dataset into training and testing sets using an 80/20 split and `random_state=42`.
6. Convert text into numerical features with `TfidfVectorizer`, fitting the vectorizer on the training text only.
7. Train and evaluate the seven classifiers.
8. Use five-fold `GridSearchCV` tuning for Random Forest, Gradient Boosting, and SVM with the parameter grids defined in the code.
9. Calculate accuracy, precision, recall, F1 score, and a confusion matrix.
10. Plot metric comparisons and, in the second version, individual confusion-matrix heatmaps.

## Evaluation Metrics

- **Accuracy:** Proportion of predictions that are correct.
- **Precision:** Proportion of predicted positive cases that are truly positive.
- **Recall:** Proportion of actual positive cases identified by the model.
- **F1 score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Counts of correct and incorrect predictions by class.

The scripts print the scores when run. No specific best-model score is claimed in this README because results depend on executing the code and are not provided as fixed, verified outputs in the repository.

## Repository Contents

```text
CLASSIFICATION-OF-CYBERBULLYING-TWEETS-USING-MACHINE-LEARNING/
├── CyberBullying Comments Dataset.csv
├── CLASSIFICATION_OF_CYBERBULLYING_TWEETS.ipynb
├── CLASSIFICATION_OF_CYBERBULLYING_TWEETS_WITH_CONFUSION_MATRIX.ipynb
├── classification_of_cyberbullying_tweets.py
├── classification_of_cyberbullying_tweets_with_confusion_matrix.py
└── archive (3).zip
```

The two notebooks contain the analysis in notebook format, while the Python files provide script versions. `archive (3).zip` is an additional packaged archive included with the repository.

## Requirements

The project uses Python and the following libraries:

- pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- scikit-learn

Install the Python packages:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn jupyter
```

The first run downloads NLTK resources for tokenization, English stop words, and WordNet lemmatization. A working internet connection may be required for these downloads.

## How to Run

### Option 1: Google Colab

1. Upload `CyberBullying Comments Dataset.csv` to your Colab session.
2. Open either notebook in Google Colab.
3. Ensure the CSV is located at `/content/sample_data/CyberBullying Comments Dataset.csv`, or update the path in the loading cell.
4. Run the notebook cell. Each supplied notebook contains the analysis in a single code cell.

### Option 2: Run locally

1. Clone or download the repository.
2. Install the dependencies listed above.
3. Open the selected Python script and change the dataset path from `/content/sample_data/CyberBullying Comments Dataset.csv` to the local path of your CSV file.
4. Run one of the scripts:

```bash
python classification_of_cyberbullying_tweets.py
```

or:

```bash
python classification_of_cyberbullying_tweets_with_confusion_matrix.py
```

## Important Limitations and Improvements

- **Text preprocessing:** The current pipeline removes punctuation, numbers, and all non-English letters. This can remove useful context, including emojis, hashtags, and spelling patterns common in social media.
- **Label interpretation:** The code treats label `1` as the positive class for precision, recall, and F1 calculations. Confirm the dataset's label definitions before interpreting results.
- **Reproducibility:** Although the train/test split uses a fixed random state, several classifiers do not set their own random seeds, so scores may vary between runs.
- **Class balance:** The supplied data is balanced, but real-world moderation data may not be. Evaluation should consider precision, recall, and class-specific errors rather than accuracy alone.
- **Potential model selection bias:** The code chooses the “best” classifier based on test-set accuracy after evaluating several models. For a stronger evaluation, use cross-validation for model selection and keep the test set untouched until final evaluation.
- **Tuning scope:** Grid search is configured for Random Forest, Gradient Boosting, and SVM only. Other classifiers use their default settings.
- **Safety and fairness:** Automated predictions can misclassify reclaimed language, dialects, quotations, or context-dependent speech. Predictions should support, not replace, careful human review in consequential moderation settings.

## Possible Extensions

- Add per-class precision, recall, and F1 scores.
- Report cross-validation results and compare against a simple baseline.
- Save the trained model and TF-IDF vectorizer for reuse.
- Test on a separate, more recent dataset to assess generalization.
- Examine false positives and false negatives by language variety and content type.

## Disclaimer

This is an educational text-classification project. It should not be treated as a complete or production-ready cyberbullying moderation system.
