import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle

#loading and balancing dataset
#loads dataset
df = pd.read_csv('file_name.csv')

print("original distribution:")
print(df['Political Lean'].value_counts())

#undersampling to balance data
from sklearn.utils import resample

dfLiberal = df[df['Political Lean'] == 'Liberal']
dfConservative = df[df['Political Lean'] == 'Conservative']

minSize = min(len(dfLiberal), len(dfConservative))

dfLiberal_downsampled = resample(dfLiberal, replace=False, n_samples=minSize, random_state=42)
dfConservative_downsampled = resample(dfConservative, replace=False, n_samples=minSize, random_state=42)

dfBalanced = pd.concat([dfLiberal_downsampled, dfConservative_downsampled])
dfBalanced = dfBalanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\ntesting to see if successfully balanced distribution:")
print(dfBalanced['Political Lean'].value_counts())

#preparing text and labels
X = dfBalanced['Title']  
y = dfBalanced['Political Lean']

#split: 80% train, 10% val, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\ntraining data size: {len(X_train)}")
print(f"validation data size: {len(X_val)}")
print(f"test data size: {len(X_test)}")

#tf-idf vectorization
#tf-idf creation
tfidf = TfidfVectorizer(max_features=5000,       #top 5000 words
    min_df=5,                 #min docs set to 5
    max_df=0.8,               #ignores words that are too common (80% docs)
    ngram_range=(1, 2),       #uses unigrams and bigrams
    stop_words='english'      #removes common english stop words
)

#fits on training data and transforms all sets
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

print(f"\nTF-IDF feature matrix shape: {X_train_tfidf.shape}")

#log-reg training
#initializes model with balanced class weights
logreg = LogisticRegression(max_iter=1000,
    random_state=42,
    class_weight='balanced',  #makes sure classes are balanced
    C=1.0                     #regularization strength
)

#training w/ log-reg
logreg.fit(X_train_tfidf, y_train)

#validation set evaluation (though this wont help us for this model since we are not fine tuning the tf-idf base model)

#predictions on validation set
y_val_pred = logreg.predict(X_val_tfidf)

print("VALIDATION SET RESULTS")

val_accuracy = accuracy_score(y_val, y_val_pred)
val_balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print("\nclassification report for validation set:")
print(classification_report(y_val, y_val_pred))

#test set evaluation
#predicts on test set
y_test_pred = logreg.predict(X_test_tfidf)

print("TEST SET RESULTS")

test_accuracy = accuracy_score(y_test, y_test_pred)
test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print("\nclassification report for test set:")
print(classification_report(y_test, y_test_pred))

#confusion matrix
#creates confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Conservative', 'Liberal'], yticklabels=['Conservative', 'Liberal'])
plt.title('confusion catrix - TF-IDF + logistic-reg')
plt.ylabel('true label')
plt.xlabel('predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix_baseline.png', dpi=300, bbox_inches='tight')
plt.show()

#analysis of notable features
#gets feature names and coefficients
featureNames = tfidf.get_feature_names_out()
coefficients = logreg.coef_[0]

#top features for each class
top_n = 20

#most liberal words (negative coefficients)
liberalIndices = np.argsort(coefficients)[:top_n]

print(f"TOP {top_n} LIBERAL FEATURES:")
for idx in liberalIndices:
    print(f"{featureNames[idx]}: {coefficients[idx]:.4f}")

#most conservative words (positive coefficients)
conservativeIndices = np.argsort(coefficients)[-top_n:][::-1]

print(f"TOP {top_n} CONSERVATIVE FEATURES:")
for idx in conservativeIndices:
    print(f"{featureNames[idx]}: {coefficients[idx]:.4f}")

#saves model and vectorizer
with open('tfidf_logreg_model.pkl', 'wb') as f:
    pickle.dump({'model': logreg, 'vectorizer': tfidf}, f)

print("\nmodel saved as 'tfidf_logreg_model.pkl'")

#inference latency measurements
#times prediction on test set
startTime = time.time()
_ = logreg.predict(X_test_tfidf)
endTime = time.time()

latency_per_sample = (endTime - startTime) / len(X_test) * 1000

print("INFERENCE LATENCY")
print(f"total test samples: {len(X_test)}")
print(f"total time: {(endTime - startTime):.4f} seconds")
print(f"latency per sample: {latency_per_sample:.4f} ms")