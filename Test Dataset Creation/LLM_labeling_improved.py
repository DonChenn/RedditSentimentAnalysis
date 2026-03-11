from google import genai
from google.genai import types
import pandas as pd
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv
import json
import uuid
import re
import typing_extensions
from collections import Counter

load_dotenv()
API_KEY = os.getenv("API_KEY")

INPUT_FILE = 'Test Dataset Creation/new_raw_reddit_comments.csv'
OUTPUT_FILE = 'Test Dataset Creation/combined_BERT_labeled_reddit_comments.csv'
DISAGREEMENT_FILE = 'Test Dataset Creation/disagreement_log.csv'
QUALITY_REPORT_FILE = 'Test Dataset Creation/quality_report.txt'

MODEL_NAME = 'gemini-2.0-flash'

# Configuration
NUM_CONSENSUS_SAMPLES = 3  # Run each comment through LLM 3 times
MIN_TOPIC_AGREEMENT = 2    # Require 2/3 agreement on topic
MIN_EMOTION_AGREEMENT = 2  # Emotion must appear in at least 2/3 samples

EMOTIONS = [
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief", "anger",
    "annoyance", "disappointment", "disapproval", "disgust", "embarrassment",
    "fear", "grief", "nervousness", "remorse", "sadness", "neutral",
    "realization", "surprise", "curiosity", "confusion"
]

TOPICS_LIST = {
    0: "INVESTIGATION: Government Investigations",
    1: "ELECTIONS: Campaigns & Primaries",
    2: "MODERATION: Community Interaction Messages",
    3: "POLITICS: Legislative Branch & Governance",
    4: "IDEOLOGY: Economic & Political Systems",
    5: "POLITICS: Ethical Misconduct & Accountability",
    6: "FOREIGN POLICY: Russia-Ukraine Conflict",
    7: "CULTURE: Values & Historical Identity",
    8: "IDEOLOGY: Extremism & Global Conflict",
    9: "LABOR: Workers Rights & Unions",
    10: "SOCIAL: Race & Civil Rights",
    11: "ECONOMY: Economics & Markets",
    12: "SOCIAL: Gender & Identity",
    13: "MEDIA: News Outlets & Propaganda",
    14: "ENVIRONMENT & IMMIGRATION: Climate & Borders",
    15: "POLITICS: Impeachment & Presidential Trials",
    16: "TECH: Social Media & Platform Rules",
    17: "ECONOMY: Wealth & Taxation",
    18: "POLITICS: Campaign Finance & Corruption",
    19: "HEALTH: Healthcare Policy & Insurance",
    20: "SOCIAL: Reproductive Rights & Abortion",
    21: "FOREIGN POLICY: Latin America & Global Policy",
    22: "POLICY: Violations & Bans",
    23: "HEALTH: COVID-19 & Public Health",
    24: "JUSTICE: Criminal Justice & Incarceration",
    25: "JUSTICE: Gun Control & 2nd Amendment",
    26: "JUSTICE: Policing & Law Enforcement",
    27: "JUSTICE: Supreme Court & Legal Rulings",
    28: "ECONOMY: Housing & Real Estate",
    29: "POLITICS: State Governance & Education",
    30: "SOCIAL: Student Debt & Higher Education",
    31: "JUSTICE: Drug Policy & Legalization",
    32: "NOISE: General Conversational Filler"
}

def setup_model():
    client = genai.Client(api_key=API_KEY)

    class AnnotationResponse(typing_extensions.TypedDict):
        emotions: list[str]
        topic_index: int
        emotion_confidence: float
        topic_confidence: float

    formatted_topics = "\n".join([f"Index {k}: {v}" for k, v in TOPICS_LIST.items()])

    # IMPROVED: Added few-shot examples
    system_instruction = f"""
You are an expert data annotator specializing in emotion and topic classification.

EXAMPLE ANNOTATIONS:

Example 1:
Post Title: "Great news for healthcare"
Comment: "I'm so proud of what we've accomplished together! This will help millions."
Analysis:
{{"emotions": ["pride", "joy"], "topic_index": 14, "emotion_confidence": 0.9, "topic_confidence": 0.95}}

Example 2:
Post Title: "Latest scandal"
Comment: "This is absolutely disgusting. I can't believe they would lie to us like this."
Analysis:
{{"emotions": ["disgust", "anger"], "topic_index": 28, "emotion_confidence": 0.95, "topic_confidence": 0.85}}

Example 3:
Post Title: "What do you think?"
Comment: "What happens next? I'm curious to see how this plays out."
Analysis:
{{"emotions": ["curiosity"], "topic_index": 17, "emotion_confidence": 0.8, "topic_confidence": 0.7}}

Example 4:
Post Title: "Economic policy discussion"
Comment: "The wealthy need to pay their fair share. It's only right."
Analysis:
{{"emotions": ["approval", "disapproval"], "topic_index": 3, "emotion_confidence": 0.85, "topic_confidence": 0.9}}

Example 5:
Post Title: "Breaking news"
Comment: "ok"
Analysis:
{{"emotions": ["neutral"], "topic_index": -1, "emotion_confidence": 0.6, "topic_confidence": 0.5}}

TASK:
Analyze Reddit comments and assign:
1) Between 1 and 3 emotions
2) Exactly ONE topic index
3) Confidence scores (0.0 to 1.0) for both

EMOTION RULES:
- Choose between 1 and 3 emotions from the list below.
- Only select multiple if the comment strongly expresses mixed feelings.
- If unsure or text is very short/unclear, choose ["neutral"].
- Return emotions as a list of strings.
- Provide emotion_confidence (how confident you are about the emotion labels).

EMOTION LIST:
{', '.join(EMOTIONS)}

TOPIC SELECTION RULES:
- You MUST consider ALL topic indices before choosing.
- Internally score EACH topic for relevance:
  0 = not relevant
  1 = weakly relevant
  2 = relevant
  3 = dominant
- Choose the topic with the highest score.
- If there is a tie, choose the LOWER index.
- If no topic scores 1 or higher, choose Index -1 (Outliers/Unclassifiable).
- Overlapping topics are expected.
- Relevance is based on the comment's MAIN CLAIM, not keyword frequency.
- Ignore incidental or passing mentions.
- Provide topic_confidence (how confident you are about the topic choice).

TOPIC LIST:
{formatted_topics}

CONFIDENCE SCORING GUIDELINES:
- 0.9-1.0: Very confident, clear and unambiguous
- 0.7-0.9: Confident, some minor ambiguity
- 0.5-0.7: Moderate confidence, multiple interpretations possible
- 0.3-0.5: Low confidence, very ambiguous
- 0.0-0.3: Very uncertain, essentially guessing

OUTPUT FORMAT:
Return ONLY valid JSON:
{{
  "emotions": ["<string>"],
  "topic_index": <integer>,
  "emotion_confidence": <float>,
  "topic_confidence": <float>
}}

Do NOT include explanations or internal scores.
"""

    config = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.9,
        max_output_tokens=512,
        response_mime_type="application/json",
        response_schema=AnnotationResponse,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ],
        system_instruction=system_instruction
    )

    return client, config

def validate_output(emotions_input, topic_idx):
    valid_emotions = [e for e in emotions_input if e in EMOTIONS]
    if not valid_emotions:
        valid_emotions = ["neutral"]

    try:
        topic_idx = int(topic_idx)
    except (ValueError, TypeError):
        topic_idx = -1

    if topic_idx not in TOPICS_LIST:
        topic_idx = -1

    return valid_emotions, topic_idx

def analyze_comment_single(client, config, title, body):
    """Single LLM call for a comment."""
    request_id = str(uuid.uuid4())
    prompt = f"RequestID: {request_id}\nPost Title: {title}\nComment Body: {body}"

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config
            )

            if not response.candidates:
                time.sleep(2)
                continue

            text = response.text.strip()
            text = re.sub(r'```json\s*|\s*```', '', text)

            result = json.loads(text)

            emotions = result.get("emotions", ["neutral"])
            topic_idx = result.get("topic_index", -1)
            emotion_conf = result.get("emotion_confidence", 0.5)
            topic_conf = result.get("topic_confidence", 0.5)

            valid_emotions, valid_topic = validate_output(emotions, topic_idx)

            return valid_emotions, valid_topic, emotion_conf, topic_conf

        except Exception as e:
            if attempt == 2:
                print(f"[Final attempt failed] Error: {e}")
            time.sleep(2)

    return ["neutral"], -1, 0.0, 0.0

# NEW: Multi-LLM consensus
def analyze_comment_with_consensus(client, config, title, body, num_samples=NUM_CONSENSUS_SAMPLES):
    """Get consensus from multiple LLM calls."""
    all_results = []

    for sample_idx in range(num_samples):
        emotions, topic, emotion_conf, topic_conf = analyze_comment_single(client, config, title, body)
        all_results.append({
            'emotions': emotions,
            'topic': topic,
            'emotion_conf': emotion_conf,
            'topic_conf': topic_conf
        })

        # Small delay between samples to avoid rate limiting
        if sample_idx < num_samples - 1:
            time.sleep(0.5)

    # Majority vote for topic
    topic_votes = [r['topic'] for r in all_results]
    topic_counter = Counter(topic_votes)
    final_topic, topic_vote_count = topic_counter.most_common(1)[0]

    # Average confidence for topic
    avg_topic_conf = sum(r['topic_conf'] for r in all_results) / num_samples

    # Collect emotions that appeared in majority of samples
    emotion_counts = {}
    for result in all_results:
        for emotion in result['emotions']:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    # Keep emotions that appeared in at least MIN_EMOTION_AGREEMENT samples
    final_emotions = [e for e, count in emotion_counts.items()
                     if count >= MIN_EMOTION_AGREEMENT]

    if not final_emotions:
        # If no consensus, take emotions from highest confidence sample
        best_sample = max(all_results, key=lambda x: x['emotion_conf'])
        final_emotions = best_sample['emotions']

    # Average confidence for emotions
    avg_emotion_conf = sum(r['emotion_conf'] for r in all_results) / num_samples

    # Calculate agreement metrics
    topic_agreement = topic_vote_count / num_samples

    return {
        'emotions': final_emotions,
        'topic': final_topic,
        'emotion_confidence': avg_emotion_conf,
        'topic_confidence': avg_topic_conf,
        'topic_agreement': topic_agreement,
        'all_topic_votes': topic_votes,
        'all_emotions': [r['emotions'] for r in all_results]
    }

# NEW: Quality validation
def validate_quality(emotions, topic_idx, title, body):
    """Flag potentially problematic labels."""
    issues = []

    # Check if text is too short for complex emotions
    text_len = len(str(title)) + len(str(body))
    if text_len < 20 and len(emotions) > 1:
        issues.append("SHORT_TEXT_MULTIPLE_EMOTIONS")

    # Check for conflicting emotions
    conflicting_pairs = [
        ({"joy", "love", "gratitude", "excitement", "optimism"},
         {"anger", "disgust", "sadness", "fear", "grief"}),
        ({"pride", "admiration"},
         {"embarrassment", "remorse", "disappointment"})
    ]

    for pos_set, neg_set in conflicting_pairs:
        if any(e in pos_set for e in emotions) and any(e in neg_set for e in emotions):
            issues.append("CONFLICTING_EMOTIONS")

    # Check if neutral appears with other strong emotions
    if "neutral" in emotions and len(emotions) > 1:
        issues.append("NEUTRAL_WITH_OTHERS")

    return issues

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Missing input file: {INPUT_FILE}")
        return

    # IMPROVED: Process all samples (removed .head(10))
    df = pd.read_csv(INPUT_FILE)
    df["row_id"] = df.index

    print(f"📊 Total samples to process: {len(df)}")

    # Resume from existing file if available
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        # Only merge columns that exist in the existing file
        merge_cols = ["row_id", "predicted_emotion", "predicted_topic"]
        for col in ["emotion_confidence", "topic_confidence", "topic_agreement"]:
            if col in existing.columns:
                merge_cols.append(col)
        df = df.merge(existing[merge_cols], on="row_id", how="left")
        # Ensure all expected columns are present
        for col in ["predicted_emotion", "predicted_topic",
                    "emotion_confidence", "topic_confidence", "topic_agreement"]:
            if col not in df.columns:
                df[col] = None
        already_done = df["predicted_emotion"].notna().sum()
        print(f"📋 Resuming: {already_done} samples already labeled")
    else:
        df["predicted_emotion"] = None
        df["predicted_topic"] = None
        df["emotion_confidence"] = None
        df["topic_confidence"] = None
        df["topic_agreement"] = None

    client, config = setup_model()
    save_counter = 0
    disagreements = []
    quality_issues = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Labeling comments"):
        if pd.notna(row["predicted_emotion"]):
            continue

        title = row["post_title"]
        body = row["comment_body"]

        # Handle empty comments
        if pd.isna(body) or str(body).strip() == "":
            df.at[i, "predicted_emotion"] = str(["neutral"])
            df.at[i, "predicted_topic"] = -1
            df.at[i, "emotion_confidence"] = 1.0
            df.at[i, "topic_confidence"] = 1.0
            df.at[i, "topic_agreement"] = 1.0
            continue

        # IMPROVED: Use consensus approach
        result = analyze_comment_with_consensus(client, config, title, body)

        emotions = result['emotions']
        topic_idx = result['topic']

        # Store results (emotions stored as string for CSV compatibility)
        df.at[i, "predicted_emotion"] = str(emotions)
        df.at[i, "predicted_topic"] = topic_idx
        df.at[i, "emotion_confidence"] = result['emotion_confidence']
        df.at[i, "topic_confidence"] = result['topic_confidence']
        df.at[i, "topic_agreement"] = result['topic_agreement']

        # NEW: Track disagreements
        if result['topic_agreement'] < 1.0:  # Not unanimous
            disagreements.append({
                'row_id': i,
                'text': f"{title} | {body}"[:200],
                'final_topic': topic_idx,
                'all_votes': result['all_topic_votes'],
                'agreement': result['topic_agreement']
            })

        # NEW: Check quality
        issues = validate_quality(emotions, topic_idx, title, body)
        if issues:
            quality_issues.append({
                'row_id': i,
                'text': f"{title} | {body}"[:200],
                'emotions': emotions,
                'topic': topic_idx,
                'issues': issues
            })

        save_counter += 1
        if save_counter >= 10:
            df.to_csv(OUTPUT_FILE, index=False)
            save_counter = 0

        time.sleep(1.0)

    # Final save
    df.to_csv(OUTPUT_FILE, index=False)

    # NEW: Save disagreement log
    if disagreements:
        disagreement_df = pd.DataFrame(disagreements)
        disagreement_df.to_csv(DISAGREEMENT_FILE, index=False)
        print(f"\n⚠️  Saved {len(disagreements)} disagreement cases to {DISAGREEMENT_FILE}")

    # NEW: Generate quality report
    with open(QUALITY_REPORT_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LABELING QUALITY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total samples processed: {len(df)}\n")
        f.write(f"Samples with disagreement: {len(disagreements)} ({len(disagreements)/len(df)*100:.1f}%)\n")
        f.write(f"Samples with quality issues: {len(quality_issues)} ({len(quality_issues)/len(df)*100:.1f}%)\n\n")

        f.write("Average Confidence Scores:\n")
        f.write(f"  Emotion Confidence: {df['emotion_confidence'].mean():.3f}\n")
        f.write(f"  Topic Confidence: {df['topic_confidence'].mean():.3f}\n")
        f.write(f"  Topic Agreement: {df['topic_agreement'].mean():.3f}\n\n")

        f.write("Low Confidence Samples (for human review):\n")
        low_conf = df[(df['emotion_confidence'] < 0.5) | (df['topic_confidence'] < 0.5)]
        f.write(f"  Total: {len(low_conf)}\n\n")

        if quality_issues:
            f.write("=" * 80 + "\n")
            f.write("QUALITY ISSUES FOUND:\n")
            f.write("=" * 80 + "\n\n")
            for issue in quality_issues[:50]:  # First 50
                f.write(f"Row {issue['row_id']}: {', '.join(issue['issues'])}\n")
                f.write(f"  Text: {issue['text']}\n")
                f.write(f"  Emotions: {issue['emotions']}\n")
                f.write(f"  Topic: {issue['topic']}\n\n")

    print(f"\n✅ Labeling complete!")
    print(f"📊 Quality report saved to {QUALITY_REPORT_FILE}")
    print(f"\n📈 Statistics:")
    print(f"   Avg Emotion Confidence: {df['emotion_confidence'].mean():.3f}")
    print(f"   Avg Topic Confidence: {df['topic_confidence'].mean():.3f}")
    print(f"   Avg Topic Agreement: {df['topic_agreement'].mean():.3f}")

if __name__ == "__main__":
    main()
