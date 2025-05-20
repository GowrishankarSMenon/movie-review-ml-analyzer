# 🎬 IMDB Sentiment Analysis – Data Processing & Cleaning

![GitHub stars](https://img.shields.io/github/stars/GowrishankarSMenon/movie-review-ml-analyzer?style=social)
![GitHub forks](https://img.shields.io/github/forks/GowrishankarSMenon/movie-review-ml-analyzer?style=social)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comprehensive data cleaning and processing pipeline for IMDB movie reviews sentiment analysis.

This project focuses on cleaning and processing a dataset of IMDB reviews for sentiment analysis purposes. The goal is to prepare a high-quality version of the dataset for training machine learning or AI models to predict sentiment (positive/negative) based on textual input.

## 📋 Table of Contents

- [Dataset](#-dataset)
- [Data Cleaning Process](#-data-cleaning-process)
- [Language Analysis](#-language-analysis)
- [Model Performance](#-model-performance)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Repository Cleanup](#-repository-cleanup)
- [License](#-license)
- [Author](#-author)

## 📊 Dataset

The project uses the popular IMDB movie reviews dataset:

- **Original Dataset**: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
  - Contains 50,000 raw IMDB movie reviews with associated sentiment labels
  - Balanced dataset with 25,000 positive and 25,000 negative reviews

> **Note**: The CSV files have been removed from version control using `git filter-repo` to reduce repository size. They can be downloaded from the link above.

![Sentiment Distribution](https://github.com/GowrishankarSMenon/movie-review-ml-analyzer/blob/main/images/positive-negative_distribution.png)
*Figure: Perfectly balanced dataset with equal numbers of positive and negative reviews*

## 🧹 Data Cleaning Process

The dataset was processed using the following steps:

1. **Text Normalization**
   - Converted all text to lowercase
   - Removed HTML tags using BeautifulSoup
   - Stripped excessive whitespace

2. **Character Cleaning**
   - Removed punctuation and special characters
   - Standardized contractions
   - Normalized unicode characters

3. **Content Filtering**
   - Removed duplicate reviews
   - Filtered out non-English content
   - Handled missing values

This makes the data ideal for natural language processing (NLP) tasks.

## 📊 Language Analysis

Analysis of word frequencies in positive and negative reviews reveals interesting patterns:

![Word Frequency Analysis](https://github.com/GowrishankarSMenon/movie-review-ml-analyzer/blob/main/images/positive_and_negative_words.png)
*Figure: Top 20 most frequent words in positive and negative reviews*

## 📈 Model Performance

The model achieved solid performance metrics on sentiment classification:

- **Accuracy**: 89.22%
- **Precision**: 0.90 (Negative), 0.88 (Positive)
- **Recall**: 0.88 (Negative), 0.91 (Positive)
- **F1-Score**: 0.89 (both classes)

### Live Predictions

The sentiment classification model accurately predicts sentiment from user input:

![Negative Review Prediction](https://github.com/GowrishankarSMenon/movie-review-ml-analyzer/blob/main/images/output-1.png)
*Figure: Model correctly predicts negative sentiment for a review*

![Positive Review Prediction](https://github.com/GowrishankarSMenon/movie-review-ml-analyzer/blob/main/images/output-2.png)
*Figure: Model correctly predicts positive sentiment for a review*

## 🚀 Usage

The cleaned dataset (`cleaned_imdb_reviews.csv`) can be used for various NLP projects:

```python
# Example code to load and explore the dataset
import pandas as pd

# Load the dataset
df = pd.read_csv('data/cleaned_imdb_reviews.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Class distribution: \n{df['sentiment'].value_counts()}")

# Preview the data
df.head()
```

### Potential Applications

- Sentiment classification models
- Text clustering and topic modeling
- Word embedding training (e.g., Word2Vec, GloVe)
- Transformer-based models (BERT, RoBERTa)

## 📁 Project Structure

```
.
├── data/
│   ├── cleaned_imdb_reviews.csv   # Cleaned data file (not in repo)
│   └── IMDB Dataset.csv           # Original dataset (not in repo)
├── images/
│   ├── output1.png                # Negative review prediction screenshot
│   ├── output2.png                # Positive review prediction screenshot
│   ├── output3.png                # Word frequency analysis chart
│   └── output4.png                # Sentiment distribution chart
├── scripts/
│   ├── clean_reviews.py           # Python script used for data cleaning
│   └── train_model.py             # Script for training the sentiment model
├── .gitignore
└── README.md
```

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GowrishankarSMenon/movie-review-ml-analyzer.git
   cd movie-review-ml-analyzer
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the `data/` directory.

## 🧰 Repository Cleanup

To remove the large CSV files from git history, the following command was used:

```bash
git filter-repo --path "IMDB Dataset.csv" --path cleaned_imdb_reviews.csv --invert-paths
```

This permanently erases them from all commits in the Git history.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✨ Author

Maintained by [Gowrishankar S Menon](https://github.com/GowrishankarSMenon).

For questions or collaboration, feel free to [open an issue](https://github.com/GowrishankarSMenon/movie-review-ml-analyzer/issues) or contact me.

---

<p align="center">
  <sub>If you found this project helpful, please consider giving it a ⭐️</sub>
</p>
