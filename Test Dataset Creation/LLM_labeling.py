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
import typing_extensions

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

TOPICS_LIST = {
    0: "Noise/Filler",
    1: "Mueller Investigation Findings",
    2: "URLs & Web Metadata",
    3: "Bill Barr Perjury Allegations",
    4: "Life & Big Questions",
    5: "Senate Hearing Debates",
    6: "Mueller Report Release & Redactions",
    7: "Russia-Ukraine Invasion & NATO",
    8: "Special Counsel Public Context",
    9: "House Oversight & Subpoenas",
    10: "Fighting for Accountability",
    11: "Fox, CNN & Media Bias",
    12: "Congress vs. The Law",
    13: "Political Talking Points & Emails",
    14: "How Politics Works",
    15: "Capitalism vs. Socialism Debate",
    16: "Reddit Community Feedback",
    17: "NY Politics & Green New Deal",
    18: "Senate Majority & McConnell",
    19: "Identity Politics & Extremism",
    20: "Leaked Letters & Documents",
    21: "Arguing over Definitions",
    22: "Socialism & Welfare",
    23: "War & Anti-War Views",
    24: "DOJ Charges & Statements",
    25: "Impeachment Trial",
    26: "Health Care & Insurance",
    27: "Campaign Finance & Economic Costs",
    28: "Primary Elections & Voter Base",
    29: "Jobs & Fair Pay",
    30: "Power & Influences",
    31: "Russia Sanctions & Collusion",
    32: "Political Skepticism",
    33: "Strong Public Reactions",
    34: "Critique of News Coverage",
    35: "Supreme Court & Judges",
    36: "Barr's Report Summary",
    37: "Race & Civil Rights History",
    38: "Approval Ratings & Polls",
    39: "Progressive & Socialist Sentiment",
    40: "Rating the Attorney General",
    41: "Taxes & The Wealthy",
    42: "Violence & Extremism",
    43: "Law Enforcement & Abortion Rights",
    44: "Calls for Resignation & Action",
    45: "Legal Rights & Lying",
    46: "Unions & Strikes",
    47: "Obstruction & Collusion",
    48: "Lying Under Oath",
    49: "China & Communism"
}

def setup_model():
    genai.configure(api_key=API_KEY)

    class AnnotationResponse(typing_extensions.TypedDict):
        emotions: list[str]
        topic_index: int

    generation_config = {
        "temperature": 0.05,
        "top_p": 0.9,
        "max_output_tokens": 512,
        "response_mime_type": "application/json",
        "response_schema": AnnotationResponse
    }

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    formatted_topics = "\n".join([f"Index {k}: {v}" for k, v in TOPICS_LIST.items()])

    system_instruction = f"""
You are an expert data annotator.

TASK:
Analyze Reddit comments and assign:
1) Between 1 and 3 emotions
2) Exactly ONE topic index

EMOTION RULES:
- Choose between 1 and 3 emotions from the list.
- Only select multiple if the comment strongly expresses mixed feelings.
- If unsure, choose ["neutral"].
- Return them as a list of strings.

TOPIC SELECTION RULES:
- You MUST consider ALL topic indices before choosing.
- Internally score EACH topic for relevance:
  0 = not relevant
  1 = weakly relevant
  2 = relevant
  3 = dominant
- Choose the topic with the highest score.
- If there is a tie, choose the LOWER index.
- If no topic scores 1 or higher, choose Index 0.
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

    if topic_idx not in TOPICS_LIST:
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
