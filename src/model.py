import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

filepath = "../data/dataset_en_clean.csv"
df = pd.read_csv(filepath)

X = df["clean_text"]
y = df["queue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=15, stratify=y)

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=15, stratify=y_train)

#print(y_train.value_counts())

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
X_validation_vec = vectorizer.transform(X_validation)

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train_vec, y_train)


model = RandomForestClassifier(class_weight='balanced')
model.fit(X_resampled, y_resampled)

print("Random Forest Test Accuracy:", model.score(X_test_vec, y_test))

y_pred = model.predict(X_test_vec)
print("Random Forest Test Results:")
print(classification_report(y_test, y_pred))

y_val_pred = model.predict(X_validation_vec)
print("Random Forest Validation Results:")
print(classification_report(y_validation, y_val_pred))


