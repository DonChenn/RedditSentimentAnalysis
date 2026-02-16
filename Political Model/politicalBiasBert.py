import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#reuses a lot of the testing code from tf-idf model (mainly baseline testing code)

#uses GPU if available (if running locally because this model is bigger and takes longer so GPU is beneficial)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

#loading and balancing dataset
#loads dataset
df = pd.read_csv('file_name.csv')

print("original distribution:")
print(df['Political Lean'].value_counts())

#undersampling to balance data
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

#loading politicalBiasBERT model
modelName = "bucketresearch/politicalBiasBERT"

tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForSequenceClassification.from_pretrained(modelName)
model = model.to(device)
model.eval()

print("politicalBiasBERT model loaded successfully")

#prediction function for politicalBiasBERT
#filters out neutral predictions (class 2) since we only have binary classification (ie: liberal or conservative)
def predictTexts(texts, batchSize=16):
    predictions = []
    indices = []
    
    #cleans up data format before feeding into tokenizer
    if hasattr(texts, 'tolist'):
        textsList = texts.tolist()
    else:
        textsList = list(texts)
    
    #processes in batches to avoid memory issues
    for i in range(0, len(textsList), batchSize):
        batchTexts = textsList[i:i + batchSize]
        
        #tokenizes
        inputs = tokenizer(batchTexts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        #moves to device
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        #gets predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batchPreds = torch.argmax(logits, dim=1).cpu().numpy()
        
        #only keeps predictions that are 0 or 1 (Liberal or Conservative), filters out class 2 (Neutral)
        for j, pred in enumerate(batchPreds):
            if pred in [0, 1]:
                predictions.append(pred)
                indices.append(i + j)
    
    #converts 0 to liberals & 1 to conservatives
    labelMap = {0: 'Liberal', 1: 'Conservative'}
    predictions = [labelMap[pred] for pred in predictions]
    
    return predictions, indices

#validation set evaluation
startTime = time.time()
y_val_pred, valIndices = predictTexts(X_val)
valTime = time.time() - startTime

#filter y_val to match the predictions (removes samples classified as neutral)
y_val_filtered = y_val.iloc[valIndices].reset_index(drop=True)

print("VALIDATION SET RESULTS")
print(f"original samples: {len(y_val)}, after filtering neutral: {len(y_val_filtered)}")

val_accuracy = accuracy_score(y_val_filtered, y_val_pred)
val_balanced_acc = balanced_accuracy_score(y_val_filtered, y_val_pred)
val_f1 = f1_score(y_val_filtered, y_val_pred, average='weighted')

print("\nclassification report for validation set:")
print(classification_report(y_val_filtered, y_val_pred))

#test set evaluation
startTime = time.time()
y_test_pred, testIndices = predictTexts(X_test)
testTime = time.time() - startTime

#filter y_test to match the predictions
y_test_filtered = y_test.iloc[testIndices].reset_index(drop=True)

print("TEST SET RESULTS")
print(f"original samples: {len(y_test)}, after filtering neutral: {len(y_test_filtered)}")

test_accuracy = accuracy_score(y_test_filtered, y_test_pred)
test_balanced_acc = balanced_accuracy_score(y_test_filtered, y_test_pred)
test_f1 = f1_score(y_test_filtered, y_test_pred, average='weighted')

print("\nclassification report for test set:")
print(classification_report(y_test_filtered, y_test_pred))

#confusion matrix
#creates confusion matrix
cm = confusion_matrix(y_test_filtered, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Conservative', 'Liberal'], yticklabels=['Conservative', 'Liberal'])
plt.title('confusion matrix for politicalBiasBERT (neutral filtered)')
plt.ylabel('true label')
plt.xlabel('predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix_politicalBERT.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nconfusion matrix saved as 'confusion_matrix_politicalBERT.png'")

#inference latency measurements
#calculates latency per sample (only for non-neutral predictions)
latency_per_sample = (testTime / len(y_test_filtered)) * 1000

print("INFERENCE LATENCY")
print(f"total test samples (after filtering): {len(y_test_filtered)}")
print(f"total time: {testTime:.4f} seconds")
print(f"latency per sample: {latency_per_sample:.4f} ms")