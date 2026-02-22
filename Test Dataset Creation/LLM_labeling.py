import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import pandas as pd
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv
import json
import uuid
import re

load_dotenv()
API_KEY = os.getenv("API_KEY")

INPUT_FILE = 'Test Dataset Creation/raw_reddit_comments.csv'
OUTPUT_FILE = 'Test Dataset Creation/TD_IF_labeled_reddit_comments.csv'

MODEL_NAME = 'gemini-2.0-flash'

EMOTIONS = [
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief", "anger",
    "annoyance", "disappointment", "disapproval", "disgust", "embarrassment",
    "fear", "grief", "nervousness", "remorse", "sadness", "neutral",
    "realization", "surprise", "curiosity", "confusion"
]

TOPICS_LIST = [
    ['market', 'money', 'capitalist', 'libertarian'],
    ['ukraine', 'crisis', 'ukraine crisis', 'invasion'],
    ['capitalism', 'anarcho', 'anarcho capitalism', 'resources'],
    ['https', 'com', 'www', 'https www'],
    ['house', 'white house', 'senate', 'congress'],
    ['white', 'white house', 'white supremacy', 'supremacy'],
    ['court', 'supreme court', 'supreme', 'justice'],
    ['new', 'new york', 'york', 'new poll'],
    ['war', 'war ukraine', 'class war', 'civil war'],
    ['socialism', 'democratic', 'democratic socialism', 'love'],
    ['putin', 'putin invasion', 'vladimir putin', 'claims'],
    ['communist', 'nepal', 'manifesto', 'communist manifesto'],
    ['workers', 'strike', 'starbucks', 'starbucks workers'],
    ['poll', 'com poll', 'view poll', 'poll https'],
    ['china', 'chinese', 'taiwan', 'africa'],
    ['news', 'fox', 'fox news', 'host'],
    ['don', 'don care', 'care', 'love'],
    ['democracy', 'social democracy', 'fight', 'economic'],
    ['russia', 'russia ukraine', 'sanctions', 'nato'],
    ['covid', 'vaccine', 'covid vaccine', 'cuba'],
    ['png', 'preview', 'width format', 'auto webp'],
    ['wage', 'minimum', 'minimum wage', 'inflation'],
    ['history', 'black', 'black history', 'black lives'],
    ['freedom', 'convoy', 'freedom convoy', 'speech'],
    ['russian', 'invasion', 'ukrainian', 'russian invasion'],
    ['socialist', 'democratic', 'democratic socialist', 'socialists'],
    ['musk', 'elon', 'elon musk', 'tesla'],
    ['anti', 'anti war', 'pro', 'anti imperialist'],
    ['police', 'officer', 'killed', 'police officer'],
    ['florida', 'gov', 'florida gov', 'governor'],
    ['class', 'working', 'working class', 'struggle'],
    ['climate', 'change', 'climate change', 'crisis'],
    ['union', 'soviet', 'soviet union', 'union address'],
    ['tax', 'rich', 'income', 'wealth'],
    ['texas', 'abortion', 'law', 'texas abortion'],
    ['rights', 'senate', 'civil', 'civil rights'],
    ['social', 'democratic', 'social democracy', 'social democratic'],
    ['race', 'theory', 'critical', 'race theory'],
    ['work', 'sex', 'life', 'future'],
    ['real', 'real estate', 'estate', 'isn'],
    ['stop', 'talking', 'cnn', 'socialists'],
    ['economy', 'economic', 'run', 'prices'],
    ['trudeau', 'act', 'canada', 'emergency'],
    ['feminist', 'radical', 'radical feminist', 'gender'],
    ['communism', 'marxist', 'kids', 'power'],
    ['jan', 'committee', 'capitol', 'jan committee'],
    ['let', 'brandon', 'congress', 'defense'],
    ['media', 'social media', 'kyle', 'app'],
    ['feminism', 'radical', 'sex', 'radical feminism'],
    ['americans', 'economically', 'millions', 'majority']
]

def setup_model():
    genai.configure(api_key=API_KEY)

    generation_config = {
        "temperature": 0.05,
        "top_p": 0.9,
        "max_output_tokens": 512,
        "response_mime_type": "application/json"
    }

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    formatted_topics = "\n".join(
        [f"Index {i}: {', '.join(t)}" for i, t in enumerate(TOPICS_LIST)]
    )

    system_instruction = f"""
You are an expert data annotator.

TASK:
Analyze Reddit comments and assign:
1) Exactly ONE emotion
2) Exactly ONE topic index

EMOTION RULES:
- Choose exactly one emotion from: {', '.join(EMOTIONS)}
- If unsure, choose "neutral"

TOPIC SELECTION RULES:
- You MUST consider ALL topic indices before choosing.
- Internally score EACH topic for relevance:
  0 = not relevant
  1 = weakly relevant
  2 = relevant
  3 = dominant
- Choose the topic with the highest score.
- If there is a tie, choose the LOWER index.
- If no topic scores 2 or higher, choose Index 0.
- Overlapping topics are expected.
- Relevance is based on the comment’s MAIN CLAIM, not keyword frequency.
- Ignore incidental or passing mentions.

TOPIC LIST:
{formatted_topics}

OUTPUT FORMAT:
Return ONLY valid JSON:
{{
  "emotion": "<string>",
  "topic_index": <integer>
}}

Do NOT include explanations or scores.
"""

    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=system_instruction
    )

def validate_output(emotion, topic_idx):
    if emotion not in EMOTIONS:
        emotion = "neutral"

    try:
        topic_idx = int(topic_idx)
    except (ValueError, TypeError):
        topic_idx = 0

    if topic_idx < 0 or topic_idx >= len(TOPICS_LIST):
        topic_idx = 0

    return emotion, topic_idx

def analyze_comment(model, title, body):
    request_id = str(uuid.uuid4())
    prompt = f"RequestID: {request_id}\nPost Title: {title}\nComment Body: {body}"

    for attempt in range(3):
        try:
            response = model.generate_content(prompt)

            if not response.parts:
                time.sleep(2)
                continue

            text = response.text.strip()
            text = re.sub(r'```json\s*|\s*```', '', text)

            result = json.loads(text)

            emotion = result.get("emotion", "neutral")
            topic_idx = result.get("topic_index", 0)

            return validate_output(emotion, topic_idx)

        except Exception as e:
            print(f"[Retry {attempt + 1}] Error: {e}")
            time.sleep(2)

    return "neutral", 0

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Missing input file: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    df["row_id"] = df.index

    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        df = df.merge(
            existing[["row_id", "predicted_emotion", "predicted_topic"]],
            on="row_id",
            how="left"
        )
    else:
        df["predicted_emotion"] = None
        df["predicted_topic"] = None

    model = setup_model()
    save_counter = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if pd.notna(row["predicted_emotion"]):
            continue

        title = row["post_title"]
        body = row["comment_body"]

        if pd.isna(body) or str(body).strip() == "":
            df.at[i, "predicted_emotion"] = "neutral"
            df.at[i, "predicted_topic"] = 0
            continue

        emotion, topic_idx = analyze_comment(model, title, body)

        df.at[i, "predicted_emotion"] = emotion
        df.at[i, "predicted_topic"] = topic_idx

        save_counter += 1
        if save_counter >= 10:
            df.to_csv(OUTPUT_FILE, index=False)
            save_counter = 0

        time.sleep(1.0)

    df.to_csv(OUTPUT_FILE, index=False)
    print("✅ Labeling complete")

if __name__ == "__main__":
    main()
