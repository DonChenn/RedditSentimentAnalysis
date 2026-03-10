import pickle
import time
import ast
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#loads final test set
final_test = pd.read_csv('final_test.csv')
final_test['comment_body'] = final_test['comment_body'].fillna('')

texts = final_test['comment_body'].tolist()

print(f"loaded {len(final_test)} test samples")
print(f"political distribution:\n{final_test['subreddit'].value_counts()}\n")

#loads trained political tfidf + logreg model
with open('tfidf_logreg_model.pkl', 'rb') as f:
    politicalBundle = pickle.load(f)
politicalVectorizer = politicalBundle['vectorizer']
politicalModel = politicalBundle['model']

#loads trained topic tfidf + nmf model
with open('tfidf_topic_model.pkl', 'rb') as f:
    topicBundle = pickle.load(f)
topicVectorizer = topicBundle['vectorizer']
nmfModel = topicBundle['nmf_model']
topicLabels = topicBundle['human_labels']

#vader base model
vader = SentimentIntensityAnalyzer()

print("models loaded\n")

#maps goEmotions labels to positive, negative, neutral buckets
#mirrors the same mapping used in GoEmotionsVADER.ipynb
sentimentMap = {
    'positive': ['admiration', 'amusement', 'approval', 'caring', 'desire', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'],
    'negative': ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'],
    'neutral':  ['neutral', 'realization', 'surprise', 'curiosity', 'confusion']
}

emotionToBucket = {}
for bucket, emotions in sentimentMap.items():
    for emotion in emotions:
        emotionToBucket[emotion] = bucket

#predicted_emotion column is a list string like ['fear'] or ['disgust', 'anger']
#extracts first emotion and maps to bucket
def parseSentimentLabel(raw):
    try:
        emotions = ast.literal_eval(raw)
        if isinstance(emotions, list) and len(emotions) > 0:
            return emotionToBucket.get(emotions[0].lower().strip(), 'neutral')
    except:
        pass
    return 'neutral'

y_true_sent = final_test['predicted_emotion'].apply(parseSentimentLabel)

#predicted_topic column is a topic index integer
#maps index to the topic label string from the nmf model
y_true_topic = final_test['predicted_topic'].apply(
    lambda idx: topicLabels[int(idx)].lower().strip() if int(idx) < len(topicLabels) else 'unknown'
)

#sentiment evaluation using vader
#maps compound score to positive, negative, neutral buckets
def vaderSentiment(text):
    score = vader.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

start = time.time()
vaderPreds = [vaderSentiment(t) for t in texts]
vaderLatency = (time.time() - start) / len(texts) * 1000

print("=" * 50)
print("SENTIMENT: VADER")
print("=" * 50)
print(classification_report(y_true_sent, vaderPreds, labels=['positive', 'negative', 'neutral']))
print(f"accuracy: {accuracy_score(y_true_sent, vaderPreds):.4f}")
print(f"f1 (weighted): {f1_score(y_true_sent, vaderPreds, average='weighted'):.4f}")
print(f"latency per sample: {vaderLatency:.4f} ms\n")

#confusion matrix for sentiment
cm_sent = confusion_matrix(y_true_sent, vaderPreds, labels=['positive', 'negative', 'neutral'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm_sent, annot=True, fmt='d', cmap='Blues',
            xticklabels=['positive', 'negative', 'neutral'],
            yticklabels=['positive', 'negative', 'neutral'])
plt.title('confusion matrix — VADER sentiment')
plt.ylabel('true label')
plt.xlabel('predicted label')
plt.tight_layout()
plt.savefig('cm_sentiment_vader.png', dpi=300)
plt.close()
print("saved cm_sentiment_vader.png")

#political evaluation using tfidf + logistic regression
start = time.time()
X_test_tfidf = politicalVectorizer.transform(texts)
politicalPreds = politicalModel.predict(X_test_tfidf)
politicalLatency = (time.time() - start) / len(texts) * 1000

print("=" * 50)
print("POLITICAL: TF-IDF + Logistic Regression")
print("=" * 50)
y_true_pol = final_test['subreddit']
print(classification_report(y_true_pol, politicalPreds, labels=['Liberal', 'Conservative']))
print(f"accuracy:      {accuracy_score(y_true_pol, politicalPreds):.4f}")
print(f"f1 (weighted): {f1_score(y_true_pol, politicalPreds, average='weighted'):.4f}")
print(f"latency per sample: {politicalLatency:.4f} ms\n")

#confusion matrix for political
cm_pol = confusion_matrix(y_true_pol, politicalPreds, labels=['Liberal', 'Conservative'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm_pol, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Liberal', 'Conservative'],
            yticklabels=['Liberal', 'Conservative'])
plt.title('confusion matrix — TF-IDF + LogReg political')
plt.ylabel('true label')
plt.xlabel('predicted label')
plt.tight_layout()
plt.savefig('cm_political_tfidf.png', dpi=300)
plt.close()
print("saved cm_political_tfidf.png")

#topic evaluation using tfidf + nmf
#assigns each post to the topic with the highest nmf activation
start = time.time()
X_test_topic = topicVectorizer.transform(texts)
topicIndices = nmfModel.transform(X_test_topic).argmax(axis=1)
topicPreds = [topicLabels[i] for i in topicIndices]
topicLatency = (time.time() - start) / len(texts) * 1000

print("=" * 50)
print("TOPIC: TF-IDF + NMF")
print("=" * 50)
topicPreds_lower = [t.lower().strip() for t in topicPreds]

#exact match system
exactAccuracy = accuracy_score(y_true_topic, topicPreds_lower)

#keyword match as secondary metric for partial credit on close predictions
def keywordMatch(pred, true):
    keywords = [k.strip() for k in pred.split(',')]
    return any(k in true for k in keywords)

keywordMatches = [keywordMatch(p, t) for p, t in zip(topicPreds_lower, y_true_topic)]
keywordAccuracy = sum(keywordMatches) / len(keywordMatches)

print(f"accuracy (exact match):   {exactAccuracy:.4f}")
print(f"accuracy (keyword match): {keywordAccuracy:.4f}")
print(f"latency per sample: {topicLatency:.4f} ms")

#samples predictions to sanity check topic output
print("\nsample predictions (first 10):")
for i in range(min(10, len(texts))):
    print(f"  true: {y_true_topic.iloc[i]}")
    print(f"  pred: {topicPreds_lower[i]}\n")

#combines all predictions into results dataframe and save
final_test['pred_sentiment'] = vaderPreds
final_test['pred_political'] = politicalPreds
final_test['pred_topic'] = topicPreds

final_test.to_csv('baseline_pipeline_results.csv', index=False)
print("full results saved to 'baseline_pipeline_results.csv'")

#summary of all three baselines
print("\n" + "=" * 50)
print("BASELINE PIPELINE SUMMARY")
print("=" * 50)
print(f"sentiment (VADER) accuracy: {accuracy_score(y_true_sent, vaderPreds):.4f}. f1: {f1_score(y_true_sent, vaderPreds, average='weighted'):.4f}. latency: {vaderLatency:.4f} ms/sample")
print(f"political (TF-IDF+LogReg) accuracy: {accuracy_score(y_true_pol, politicalPreds):.4f}. f1: {f1_score(y_true_pol, politicalPreds, average='weighted'):.4f}. latency: {politicalLatency:.4f} ms/sample")
print(f"topic (TF-IDF+NMF) exact: {exactAccuracy:.4f}. keyword: {keywordAccuracy:.4f}. latency: {topicLatency:.4f} ms/sample")

#combined scoring using partial credit system
#0 correct = 0%, 1 correct = 33.33%, 2 correct = 66.67%, 3 correct = 100%
scoreMap = {0: 0, 1: 33.33, 2: 66.67, 3: 100}

sentCorrect = (y_true_sent.values == vaderPreds)
polCorrect = (y_true_pol.values == politicalPreds)
topicCorrect = (y_true_topic.values == topicPreds_lower)

#counts how many models got each sample right
numCorrect = sentCorrect.astype(int) + polCorrect.astype(int) + topicCorrect.astype(int)

#applies partial credit scores per sample then average
sampleScores = [scoreMap[n] for n in numCorrect]
combinedScore = np.mean(sampleScores)

#counts how many samples fell into each category
counts = {
    'all 3 correct\n(100%)': int((numCorrect == 3).sum()),
    '2 correct\n(66.67%)': int((numCorrect == 2).sum()),
    '1 correct\n(33.33%)': int((numCorrect == 1).sum()),
    '0 correct\n(0%)': int((numCorrect == 0).sum()),
}

print("\n")
print("COMBINED BASELINE SCORE (PARTIAL CREDIT)")
print(f"overall combined score: {combinedScore:.2f}%")
print(f"\nbreakdown:")
for label, count in counts.items():
    pct = count / len(numCorrect) * 100
    print(f"  {label.replace(chr(10), ' ')}: {count} samples ({pct:.1f}%)")

#horizontal bar chart for individual model performance
fig, ax = plt.subplots(figsize=(8, 4))

modelNames = ['Topic', 'Political', 'Sentiment']
modelScores = [exactAccuracy, accuracy_score(y_true_pol, politicalPreds), accuracy_score(y_true_sent, vaderPreds)]
modelColors = ['#2ecc71', '#5bc8c8', '#f08080']

bars = ax.barh(modelNames, modelScores, color=modelColors, edgecolor='white')

#adds score labels inside each bar
for bar, score in zip(bars, modelScores):
    ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
            f'{score:.3f}', va='center', ha='right', fontsize=11, color='white', fontweight='bold')

#adds dashed line at 0.5 as reference
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
ax.set_xlim(0, 1.0)
ax.set_xlabel('accuracy', fontsize=11)
ax.set_title('Model Performance', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300)
plt.close()
print("saved model_performance.png")

#vertical bar chart for combined partial credit breakdown
fig, ax = plt.subplots(figsize=(8, 5))

barLabels = list(counts.keys())
barValues = list(counts.values())
barPcts = [v / len(numCorrect) * 100 for v in barValues]
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

bars = ax.bar(barLabels, barPcts, color=colors, edgecolor='white', linewidth=1.2)

#adds count and percentage labels on top of each bar
for bar, count, pct in zip(bars, barValues, barPcts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f'{count} samples\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10)

ax.set_ylabel('% of samples', fontsize=12)
ax.set_title(f'baseline pipeline combined partial credit score: {combinedScore:.1f}%', fontsize=13)
ax.set_ylim(0, max(barPcts) + 15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('combined_score_breakdown.png', dpi=300)
plt.close()
print("saved combined_score_breakdown.png")