import pandas as pd
from google import genai
from google.genai import errors
import time
import os

# 1. Initialize the Gemini Client
client = genai.Client(api_key="AIzaSyAKEN5HCbfrgI6YYcH9rxA_aTweUiulUpw")
MODEL_NAME = 'gemini-2.5-flash'

# The 28 GoEmotions labels
EMOTIONS = [
    "admiration", "amusement", "approval", "caring", "desire", "excitement", 
    "gratitude", "joy", "love", "optimism", "pride", "relief", "anger", 
    "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", 
    "fear", "grief", "nervousness", "remorse", "sadness", "neutral", 
    "realization", "surprise", "curiosity", "confusion"
]

def classify_comment_with_retry(comment_text, retries=3):
    """Classifies a comment with error handling and retry logic."""
    prompt = f"""
    Analyze the following Reddit comment and provide:
    1. One emotion from this list: {EMOTIONS}
    2. A short political topic label (1-3 words).
    Format the response strictly as: Emotion | Topic
    Comment: "{comment_text}"
    """
    
    for i in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, 
                contents=prompt
            )
            
            if response.text:
                parts = response.text.strip().split('|')
                emotion = parts[0].strip().lower()
                topic = parts[1].strip() if len(parts) > 1 else "Unknown"
                
                # Validation
                if emotion not in EMOTIONS:
                    emotion = "neutral"
                return emotion, topic
            return "neutral", "Unknown"
            
        except errors.APIError as e:
            if "429" in str(e):
                wait_time = 30 
                print(f"Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"API Error: {e}")
                return "error", "error"
    return "timeout", "timeout"

def main():
    # Connection Test
    print(f"Verifying connection to {MODEL_NAME}...")
    try:
        test = client.models.generate_content(model=MODEL_NAME, contents="hi")
        print(f"Connection Successful! Response: {test.text.strip()}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not connect to {MODEL_NAME}. Check your API Key and Quota. \nDetail: {e}")
        return

    input_file = 'raw_reddit_comments.csv'
    output_file = 'labeled_reddit_comments.csv'
    
    # Load data or resume progress
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"Resuming progress from {output_file}...")
    else:
        df = pd.read_csv(input_file)
        df['sentiment'] = None
        df['topic'] = None

    remaining_indices = df[df['sentiment'].isna()].index
    print(f"Processing {len(remaining_indices)} remaining comments...")

    for count, index in enumerate(remaining_indices):
        comment = df.at[index, 'comment_body']
        
        emotion, topic = classify_comment_with_retry(comment)
        
        df.at[index, 'sentiment'] = emotion
        df.at[index, 'topic'] = topic
        
        # Periodic save (checkpoint)
        if (count + 1) % 10 == 0:
            df.to_csv(output_file, index=False)
            print(f"Processed {count + 1} comments. Checkpoint saved.")
        
        # Pacing to stay under Free Tier RPM
        time.sleep(4)

    # Final save
    df.to_csv(output_file, index=False)
    print(f"Success! All comments processed and saved to {output_file}")

if __name__ == "__main__":
    main()