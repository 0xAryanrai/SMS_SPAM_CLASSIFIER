# SMS_SPAM_CLASSIFIER
A simple sms spam/ham classifer using Multinomial Naive Bayes.

# SMS Spam Classifier

A machine learning project that classifies SMS text messages as **spam** or **ham** (not spam) using classical NLP preprocessing and Naive Bayes models.

## Overview

This notebook walks through a complete text classification pipeline: cleaning raw SMS data, exploring it, transforming the text into numerical features, and training/evaluating multiple Naive Bayes classifiers to find the best-performing model.

## Dataset

- **File:** `spam.csv`
- **Columns used:** `v1` (label: `ham`/`spam`), `v2` (raw SMS text)
- Three unnamed columns with mostly null values are dropped during cleaning.

## Pipeline

1. **Data Cleaning**
   - Dropped unused/null columns
   - Renamed columns to `target` and `text`
   - Label-encoded `target` (`ham` → 0, `spam` → 1)
   - Checked for missing values and removed duplicate rows

2. **Exploratory Data Analysis (EDA)**
   - Class distribution (spam vs. ham) visualized with a pie chart
   - Engineered features: `num_character`, `num_words`, `num_sentence` (via NLTK tokenization)
   - Statistical comparison of these features between spam and ham messages
   - Visualized relationships with histograms, pairplots, and a correlation heatmap

3. **Text Preprocessing**
   - Lowercasing
   - Tokenization
   - Removal of special characters, punctuation, and stopwords
   - Stemming (Porter Stemmer)
   - Combined into a `transform_text()` function applied to create a `transformed_text` column
   - Word clouds generated to visualize the most frequent words in spam vs. ham messages

4. **Feature Extraction**
   - Bag-of-Words vectorization using `CountVectorizer`

5. **Model Building**
   - Train/test split (80/20)
   - Trained three Naive Bayes variants:
     - `GaussianNB`
     - `BernoulliNB`
     - `MultinomialNB`

6. **Evaluation**

   | Model | Accuracy | Precision |
   |---|---|---|
   | GaussianNB | 0.874 | 0.535 |
   | BernoulliNB | 0.970 | 0.992 |
   | MultinomialNB | 0.971 | 0.904 |

   **BernoulliNB** achieved the best precision, making it well-suited for spam detection where minimizing false positives (ham misclassified as spam) is important.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (LabelEncoder, CountVectorizer, Naive Bayes models, train/test split, metrics)
- NLTK (tokenization, stopwords, stemming)
- matplotlib, seaborn (visualization)
- WordCloud (text visualization)

## Requirements

```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
wordcloud
```

Also run once to fetch NLTK resources:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

## Next Steps (Planned)

- Further model improvement based on evaluation metrics
- Build a simple web interface for the classifier
- Deploy the web app

## Usage

1. Place `spam.csv` in the same directory as the notebook.
2. Run all cells in order.
3. The trained models predict whether an input SMS is spam or ham.
