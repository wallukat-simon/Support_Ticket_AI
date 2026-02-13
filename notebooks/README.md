# Support Ticket Classification 
#### Project Overview
The goal of this project is to build a machine learning system that automatically classifies customer support tickets into predefined categories based on their textual content.

In Phase 1, the dataset is explored and preprocessed to create a clean and structured input for subsequent machine learning models.

#### Dataset
The dataset consists of customer support tickets with the following relevant fields:
- `title`: short summary of the issue  
- `body`: detailed description of the issue  
- `label`: ticket category (e.g. billing, technical support, account)  
- `priority`: urgency level (low, medium, high)


## Phase 1: Data Exploration & Preprocessing

#### Exploratory Data Analysis
The following aspects were analyzed:
- Distribution of ticket categories  
- Distribution of priority levels  
- Ticket length (measured as number of words)  
- Ticket length per category and per priority  

The analysis revealed a strong class imbalance and a wide variation in ticket lengths, reflecting realistic customer support data.

#### Text Preprocessing
To prepare the text data for machine learning, the following preprocessing steps were applied:
- Combination of ticket title and body into a single text field  
- Lowercasing  
- Removal of special characters and formatting artifacts  
- Tokenization  
- Stopword removal  
- Optional lemmatization  

After preprocessing, each ticket is represented by a cleaned text field containing mainly content-bearing words.

#### Output of Phase 1
The result of this phase is a cleaned dataset that can be directly used for model training in Phase 2.

Output file:
- `clean_data.csv`

This file contains:
- `clean_text`
- `label`
- `priority`

---

## Phase 2: Model Training and Evaluation

In this phase, the cleaned dataset from Phase 1 is used to train and evaluate machine learning models for support ticket classification.

The data is split into training, validation, and test sets using stratified sampling to preserve the original class distribution. Text data is transformed into numerical feature vectors using TF-IDF.

Due to a strong class imbalance, different balancing strategies were evaluated. Oversampling of minority classes resulted in the best performance and was therefore applied to the training set.

Several classification models were considered, and Random Forest was selected based on validation performance and robustness. Model performance is evaluated using a confusion matrix and a classification report.

Hyperparameter tuning is performed using GridSearchCV. The tuning process leads to only marginal improvements, indicating that classical machine learning methods may have reached their performance limit for this task.

In the next phase, deep learning-based approaches will be explored to further improve classification performance.

---

## Phase 3: Deep Learning Approaches
In Phase 3, deep learning models are applied to the ticket classification task in order to better capture semantic relationships in the text data.

An LSTM was chosen as the baseline deep learning model because it explicitly models word order and long-range dependencies in text, providing a more expressive representation than traditional bag-of-words approaches while remaining computationally efficient and interpretable.

The deep learning workflow implemented in [notebooks/deep_learning.ipynb](notebooks/deep_learning.ipynb) includes:
- Loading the cleaned dataset (`dataset_en_clean.csv`) and encoding the target labels.
- Tokenizing text with a whitespace tokenizer and building a vocabulary of the most frequent words.
- Converting tickets to integer sequences and padding/truncating to a fixed length (80 tokens).
- Creating a stratified train/test split and wrapping data in a custom PyTorch `Dataset` and `DataLoader`.
- Defining an LSTM classifier with embeddings, a (bi-directional) LSTM encoder, dropout, and a linear classification head.
- Handling class imbalance with weighted cross-entropy loss and training for 75 epochs with gradient clipping.
- Evaluating performance using a classification report and saving the trained model state to `lstm_classifier_state.pt`.

Results:
- The LSTM-based model did not outperform the classical TF-IDF baseline. The most likely contributors are the limited dataset size, class imbalance, and the strong performance of keyword-based TF-IDF features for this task.