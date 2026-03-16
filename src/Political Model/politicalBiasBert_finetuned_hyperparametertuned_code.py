import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

#finetuning politicalBiasBERT on our liberal vs conservative dataset
#builds off zero shot baseline: same model but now trained on our data

#uses GPU if available
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
X = dfBalanced['Title'] + ' ' + dfBalanced['Text'].fillna('')
y = dfBalanced['Political Lean']

#converts string labels to integers for training (Liberal=0, Conservative=1)
labelMap = {'Liberal': 0, 'Conservative': 1}
reverseLabelMap = {0: 'Liberal', 1: 'Conservative'}
yEncoded = y.map(labelMap)

#split: 80% train, 10% val, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, yEncoded, test_size=0.2, random_state=42, stratify=yEncoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\ntraining data size: {len(X_train)}")
print(f"validation data size: {len(X_val)}")
print(f"test data size: {len(X_test)}")

#loading politicalBiasBERT model
#original model has 3 output classes (Liberal, Neutral, Conservative)
#replacing the classification head with a new 2 class head for binary classification
modelName = "bucketresearch/politicalBiasBERT"

tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForSequenceClassification.from_pretrained(modelName, num_labels=2, ignore_mismatched_sizes=True, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2) #doubled dropout
model = model.to(device)

print("politicalBiasBERT loaded with new 2-class classification head")

#dataset class for tokenizing and batching reddit posts
class RedditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, maxLength=512): #doubled max length
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        self.tokenizer = tokenizer
        self.maxLength = maxLength

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.maxLength,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

#creating datasets and dataloaders
trainDataset = RedditDataset(X_train, y_train, tokenizer)
valDataset = RedditDataset(X_val, y_val, tokenizer)
testDataset = RedditDataset(X_test, y_test, tokenizer)

trainLoader = DataLoader(trainDataset, batch_size=16, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=16, shuffle=False)
testLoader = DataLoader(testDataset, batch_size=16, shuffle=False)

#training setup
#uses linear warmup scheduler with adamw optimizer
numEpochs = 5 #inreased numEpochs to 5 from 3
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01) #made learning rate faster to learn quicker and more efficiently
#weight decay should stay same, changing only made performance worse
totalSteps = len(trainLoader) * numEpochs
#linear was better than cosine scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=totalSteps // 20, num_training_steps=totalSteps) #changing numsteps
#training loop
#saves best model checkpoint based on validation balanced accuracy
print("\nstarting fine tuning...")

bestValAcc = 0
bestModelPath = 'politicalBiasBERT_finetuned_best.pt'

for epoch in range(numEpochs):
    #training phase
    model.train()
    totalLoss = 0
    epochStart = time.time()

    for batchIdx, batch in enumerate(trainLoader):
        inputIds = batch['input_ids'].to(device)
        attentionMask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=inputIds, attention_mask=attentionMask, labels=labels)
        loss = outputs.loss
        totalLoss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1) #gradient clipping tested, best at orig 1
        optimizer.step()
        scheduler.step()

        #prints progress every 50 batches
        if (batchIdx + 1) % 50 == 0:
            print(f"epoch {epoch+1}/{numEpochs} | batch {batchIdx+1}/{len(trainLoader)} | loss: {loss.item():.4f}")

    avgLoss = totalLoss / len(trainLoader)
    epochTime = time.time() - epochStart

    #validation phase
    model.eval()
    valPreds = []
    valTrue = []

    with torch.no_grad():
        for batch in valLoader:
            inputIds = batch['input_ids'].to(device)
            attentionMask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=inputIds, attention_mask=attentionMask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            valPreds.extend(preds)
            valTrue.extend(labels.cpu().numpy())

    valAcc = balanced_accuracy_score(valTrue, valPreds)
    print(f"\nepoch {epoch+1}/{numEpochs} complete. avg loss is: {avgLoss:.4f}. val balanced acc: {valAcc:.4f}. time taken: {epochTime:.1f}s")

    #saves checkpoint if best validation accuracy so far
    if valAcc > bestValAcc:
        bestValAcc = valAcc
        torch.save(model.state_dict(), bestModelPath)
        print(f"new best model saved (val balanced acc: {bestValAcc:.4f})")

#loads best model for final evaluation
print(f"\nloading best model (val balanced acc: {bestValAcc:.4f})")
model.load_state_dict(torch.load(bestModelPath))
model.eval()

#prediction function for fine tuned politicalBiasBERT
#no neutral filtering needed since model now has 2 class head
def predictTexts(texts, batchSize=16):
    predictions = []

    #cleans up data format before feeding into tokenizer
    if hasattr(texts, 'tolist'):
        textsList = texts.tolist()
    else:
        textsList = list(texts)

    #processes in batches to avoid memory issues
    for i in range(0, len(textsList), batchSize):
        batchTexts = textsList[i:i + batchSize]

        inputs = tokenizer(batchTexts, padding=True, truncation=True, max_length=512, return_tensors="pt") #doubled max length
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        #converts 0 to Liberal & 1 to Conservative
        predictions.extend([reverseLabelMap[p] for p in preds])

    return predictions

#validation set evaluation
startTime = time.time()
y_val_pred = predictTexts(X_val)
valTime = time.time() - startTime

#converts integer labels back to strings for classification report
y_val_str = y_val.map(reverseLabelMap)

print("VALIDATION SET RESULTS")
print(f"original samples: {len(y_val)}")

print("\nclassification report for validation set:")
print(classification_report(y_val_str, y_val_pred))

#test set evaluation
startTime = time.time()
y_test_pred = predictTexts(X_test)
testTime = time.time() - startTime

#converts integer labels back to strings for classification report
y_test_str = y_test.map(reverseLabelMap)

print("TEST SET RESULTS")
print(f"original samples: {len(y_test)}")

print("\nclassification report for test set:")
print(classification_report(y_test_str, y_test_pred))

#confusion matrix
#creates confusion matrix
cm = confusion_matrix(y_test_str, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Conservative', 'Liberal'], yticklabels=['Conservative', 'Liberal'])
plt.title('confusion matrix for politicalBiasBERT (fine tuned)')
plt.ylabel('true label')
plt.xlabel('predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix_politicalBERT_finetuned.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nconfusion matrix saved as 'confusion_matrix_politicalBERT_finetuned.png'")

#inference latency measurements
#calculates latency per sample
latency_per_sample = (testTime / len(X_test)) * 1000

print("INFERENCE LATENCY")
print(f"total test samples: {len(X_test)}")
print(f"total time: {testTime:.4f} seconds")
print(f"latency per sample: {latency_per_sample:.4f} ms")