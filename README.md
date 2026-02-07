# Support Ticket Classification – Phase 1: Data Exploration & Preprocessing

## Project Overview
The goal of this project is to build a machine learning system that automatically classifies customer support tickets into predefined categories based on their textual content.

In Phase 1, the dataset is explored and preprocessed to create a clean and structured input for subsequent machine learning models.

---

## Dataset
The dataset consists of customer support tickets with the following relevant fields:
- `title`: short summary of the issue  
- `body`: detailed description of the issue  
- `label`: ticket category (e.g. billing, technical support, account)  
- `priority`: urgency level (low, medium, high)

---

## Exploratory Data Analysis
The following aspects were analyzed:
- Distribution of ticket categories  
- Distribution of priority levels  
- Ticket length (measured as number of words)  
- Ticket length per category and per priority  

The analysis revealed a strong class imbalance and a wide variation in ticket lengths, reflecting realistic customer support data.

---

## Text Preprocessing
To prepare the text data for machine learning, the following preprocessing steps were applied:
- Combination of ticket title and body into a single text field  
- Lowercasing  
- Removal of special characters and formatting artifacts  
- Tokenization  
- Stopword removal  
- Optional lemmatization  

After preprocessing, each ticket is represented by a cleaned text field containing mainly content-bearing words.

---

## Output of Phase 1
The result of this phase is a cleaned dataset that can be directly used for model training in Phase 2.

Output file:
- `clean_data.csv`

This file contains:
- `clean_text`
- `label`
- `priority`

---

## Next Steps (Phase 2)
In the next phase, machine learning models will be trained using the cleaned dataset. This includes:
- Train/test split  
- Feature extraction using TF-IDF  
- Training and evaluation of classification models  
- Analysis of class imbalance effects
