# Sentiment Analysis Using Naive Bayes

**Author:** Sunny Mallick  
**Roll No:** 2101207  
**Course:** CS683 - NLP, Autumn Semester 2024-25

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Approach](#approach)
- [Model Results](#model-results)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Running the Code](#running-the-code)
- [Evaluation Metrics](#evaluation-metrics)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction

This project performs sentiment analysis on movie reviews using a Naive Bayes classifier. The goal is to classify reviews as either positive or negative, based on the words they contain. The implementation uses Add-1 (Laplace) smoothing to handle words not seen during training.

## Dataset Description

- **Dataset:** Movie reviews from the *rt-polaritydata* dataset
- **Positive reviews:** 5331
- **Negative reviews:** 5331
- **Preprocessing:**
  - Tokenization
  - Stopword removal
- **Train/Validation/Test Split:** 80% / 10% / 10%

## Approach

1. **Data Preprocessing:** Tokenization and stopword removal.
2. **Naive Bayes Training:** The classifier was trained using Laplace smoothing.
3. **Prediction:** Sentiment classification based on the maximum likelihood of words.
4. **Evaluation:** The model was evaluated using standard metrics like Accuracy, Precision, Recall, and F1-Score.

## Model Results

| Metric            | Value  |
|-------------------|--------|
| True Positives (TP)| 635    |
| True Negatives (TN)| 666    |
| False Positives (FP)| 165   |
| False Negatives (FN)| 196   |
| Accuracy           | 78.28% |
| Precision          | 79%    |
| Recall             | 76%    |
| F1-Score           | 78%    |

## Software Requirements

- **Python 3.x**
- **Libraries:**
  - `re` (Regular Expressions)
  - `math`
  - `collections.defaultdict`
- **Dataset:** *rt-polaritydata* (included in the repository)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/naive-bayes-sentiment-analysis.git
   ```

2. Navigate to the project directory:

   ```bash
   cd naive-bayes-sentiment-analysis
   ```

3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the sentiment analysis script:

   ```bash
   python src/sentimentAnalysis.py
   ```

This will preprocess the data, train the Naive Bayes model, and output the evaluation results, including accuracy, precision, recall, and F1-score.

## Evaluation Metrics

The model performance is evaluated using the following metrics:
- **Accuracy:** The proportion of correctly classified reviews.
- **Precision:** The ratio of correctly predicted positive reviews to total predicted positives.
- **Recall:** The ratio of correctly predicted positive reviews to all actual positives.
- **F1-Score:** The harmonic mean of precision and recall.

## Conclusion

The Naive Bayes model with Laplace smoothing achieved a reasonable performance on the sentiment analysis task, with a good balance between precision, recall, and F1-score. Future improvements could involve more sophisticated text preprocessing techniques and experimenting with other machine learning models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Updated File Structure:

```
naive-bayes-sentiment-analysis/
│
├── dataset/
│   ├── rt-polarity.pos             # Positive movie reviews dataset
│   ├── rt-polarity.neg             # Negative movie reviews dataset
│
├── src/
│   └── sentimentAnalysis.py        # Main script for sentiment analysis using Naive Bayes
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── LICENSE                         # License file (e.g., MIT License)
```
