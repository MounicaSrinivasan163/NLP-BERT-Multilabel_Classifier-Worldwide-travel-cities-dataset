# 🌍 Travel Destination Multi-Label Classification

## Project Overview
This project classifies travel destinations based on their **short descriptions** into multiple categories such as culture, adventure, nature, beaches, nightlife, cuisine, wellness, urban, and seclusion.  

We explore **BERT embeddings + Logistic Regression** approaches for **multi-label classification**. The goal is to automatically assign labels to destinations for recommendation, search, or analytics.

---

## Dataset
The dataset contains **560 travel destinations** with rich descriptive text and ratings across multiple features.

### Columns

| Column | Description |
|--------|-------------|
| `id` | Unique identifier for each destination |
| `city` | Name of the city |
| `country` | Country of the destination |
| `region` | Region (e.g., Europe, Asia) |
| `short_description` | Short textual description of the destination |
| `latitude` | Geographic latitude |
| `longitude` | Geographic longitude |
| `avg_temp_monthly` | Average monthly temperature (object) |
| `ideal_durations` | Recommended visit duration (object) |
| `budget_level` | Travel budget category (object) |
| `culture` | Rating (1–5) for cultural attractions |
| `adventure` | Rating (1–5) for adventure activities |
| `nature` | Rating (1–5) for natural attractions |
| `beaches` | Rating (1–5) for beaches |
| `nightlife` | Rating (1–5) for nightlife |
| `cuisine` | Rating (1–5) for cuisine |
| `wellness` | Rating (1–5) for wellness/spa |
| `urban` | Rating (1–5) for urban experience |
| `seclusion` | Rating (1–5) for secluded/quiet areas |

> **Note:** Ratings ≥4 are considered **relevant** for multi-label classification.

---

## Data Pipeline

### 1️⃣ Preprocessing
- Convert rating columns into **binary labels**: 1 if rating ≥ 4, else 0.  
- Clean text (optional: lowercasing, remove punctuation, stopwords).  
- Split dataset into **train/test** sets (80/20).  

### 2️⃣ Feature Engineering
- **BERT:** Use `Sentence-BERT` (`all-MiniLM-L6-v2`) to encode descriptions into **dense 384-dimensional embeddings**.

### 3️⃣ Multi-Label Classification
- **BERT + Logistic Regression:** Use `OneVsRestClassifier(LogisticRegression)` for multi-label predictions.

### 4️⃣ Evaluation
- **Per-label metrics:** Precision, Recall, F1-score.  
- **Overall metrics:** Hamming Loss, F1-score (micro and macro averages).  
- Compare model performance between **Naïve Bayes** and **BERT embeddings**.

### 5️⃣ Prediction
- Input a new destination description.  
- Generate BERT embeddings.  
- Predict relevant labels (e.g., `culture=1`, `beaches=1`, `nightlife=1`).

---

## Technologies & Libraries
- Python 3.12  
- pandas, numpy  
- scikit-learn  
- sentence-transformers (`all-MiniLM-L6-v2`)  
- Hugging Face Transformers  
- Jupyter Notebook / Google Colab  

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/travel-multilabel-classification.git
cd travel-multilabel-classification
