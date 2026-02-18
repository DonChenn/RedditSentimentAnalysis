import google.generativeai as genai
import pandas as pd
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

INPUT_FILE = 'raw_reddit_comments.csv'
OUTPUT_FILE = 'labeled_reddit_comments.csv'
MODEL_NAME = 'gemini-2.5-flash'

EMOTIONS = [
    "admiration", "amusement", "approval", "caring", "desire", "excitement", 
    "gratitude", "joy", "love", "optimism", "pride", "relief", "anger", 
    "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", 
    "fear", "grief", "nervousness", "remorse", "sadness", "neutral", 
    "realization", "surprise", "curiosity", "confusion"
]

def setup_model():
    genai.configure(api_key=API_KEY)
    
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 1024,
        "response_mime_type": "application/json",
    }
    
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        system_instruction=(
            "You are an expert data annotator. Your task is to analyze Reddit comments "
            "for sentiment and political topic. You will be given a Post Title and a Comment Body. "
            f"You must select exactly one emotion from this list: {', '.join(EMOTIONS)}. "
            "You must also identify the primary specific political topic (e.g., 'Immigration', 'NATO', 'Healthcare')."
        )
    )
    return model

def analyze_comment(model, title, body):
    prompt = f"""
    Please analyze the following Reddit comment:
    
    **Post Title:** {title}
    **Comment Body:** {body}
    
    Return a JSON object with these two keys:
    1. "emotion": The single most appropriate label from the GoEmotions list.
    2. "topic": A concise political topic string (1 word only).
    """
    
    try:
        response = model.generate_content(prompt)
        import json
        result = json.loads(response.text)
        return result.get("emotion"), result.get("topic")
    except Exception as e:
        print(f"Error processing row: {e}")
        return None, None

def main():
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")

    # 2. Setup output dataframe (resume if file exists)
    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming from {OUTPUT_FILE}...")
        results_df = pd.read_csv(OUTPUT_FILE)
        processed_indices = set(results_df.index)
    else:
        results_df = df.copy()
        results_df['predicted_emotion'] = None
        results_df['predicted_topic'] = None
        processed_indices = set()

    # 3. Initialize Model
    model = setup_model()

    # 4. Iterate and Label
    for index, row in tqdm(results_df.iterrows(), total=len(results_df)):
        if index in processed_indices and pd.notna(results_df.at[index, 'predicted_emotion']):
            continue

        title = row['post_title']
        body = row['comment_body']
        
        # Skip empty comments
        if pd.isna(body) or str(body).strip() == "":
            continue

        emotion, topic = analyze_comment(model, title, body)
        
        results_df.at[index, 'predicted_emotion'] = emotion
        results_df.at[index, 'predicted_topic'] = topic

        time.sleep(0.1) 
        
        # Save every 10 rows to prevent data loss
        if index % 10 == 0:
            results_df.to_csv(OUTPUT_FILE, index=False)

    # Final save
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
