import google.generativeai as genai
import pandas as pd
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv
import json

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

TOPICS_LIST = [
    "Other, Irrelevant, Noise",
    "ukraine, russia, russian, putin",
    "trump, desantis, election, to",
    "the, of, and, to",
    "women, her, she, men",
    "the, to, police, of",
    "covid, vaccine, insurance, healthcare",
    "podcast, this, meme, you",
    "the, of, and, war",
    "democracy, you, to, social",
    "workers, wage, minimum, unions",
    "tax, debt, student, poverty",
    "biden, bidens, joe, poll",
    "cuba, venezuela, cuban, us",
    "inflation, bitcoin, crypto, fed",
    "facebook, truth, social, app",
    "china, chinese, north, korea",
    "climate, change, carbon, woke",
    "abortion, texas, law, roe",
    "land, property, housing, rent",
    "tucker, manchin, carlson, joe",
    "elon, musk, tesla, musks",
    "the, party, catalan, and",
    "labour, kshama, sawant, recall",
    "market, monopolies, public, free",
    "mask, masks, mandates, mandate",
    "schools, education, books, school",
    "nepal, maoists, maoist, india",
    "amazon, kelloggs, workers, deere",
    "prosecutors, resign, manhattan, cuomo",
    "refugees, australian, australia, png",
    "edward, snowden, humanitarian, ccr",
    "ancap, ancaps, proudhon, nap",
    "dsa, bowman, dsas, expel",
    "latino, california, term, voters",
    "kyle, rittenhouse, george, floyd",
    "hahah, aocia, aged, these",
    "whatever, unrelated, weekly, socdem",
    "freedom, person, cancel, culture",
    "marijuana, legalization, cannabis, tobacco",
    "sowell, thomas, equality, fairness",
    "marjorie, taylor, greene, nationalist",
    "brandon, lets, go, fritolay",
    "harris, kamala, bomb, box",
    "matt, gaetz, gaetzs, exgirlfriend",
    "flag, honk, honking, mascots",
    "breyer, stephen, retire, justice",
    "irish, ireland, loyalists, nationalists",
    "harry, reid, obama, barack",
    "peace, trump, prize, president"
]

def setup_model():
    genai.configure(api_key=API_KEY)
    
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 1024,
        "response_mime_type": "application/json",
    }
    
    formatted_topics = "\n".join([f"Index {i}: {t}" for i, t in enumerate(TOPICS_LIST)])

    system_instruction = (
        "You are an expert data annotator. Your task is to analyze Reddit comments "
        "for sentiment and political topic.\n"
        f"1. EMOTION: Select exactly one emotion from: {', '.join(EMOTIONS)}.\n"
        f"2. TOPIC: Select exactly one Topic Index from the following list:\n{formatted_topics}\n"
        "Return JSON."
    )
    
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        system_instruction=system_instruction
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
        result = json.loads(response.text)
        
        emotion = result.get("emotion")
        topic_idx = result.get("topic_index")
        
        if isinstance(topic_idx, int) and 0 <= topic_idx < len(TOPICS_LIST):
            return emotion, topic_idx
        else:
            return emotion, -1
        
    except Exception as e:
        print(f"Error processing row: {e}")
        return "ERROR", "ERROR"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. Load FULL Input Data
    df_input = pd.read_csv(INPUT_FILE)
    print(f"Input file has {len(df_input)} rows.")

    # 2. Handle Resume Logic Correctly
    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming from {OUTPUT_FILE}...")
        df_existing = pd.read_csv(OUTPUT_FILE)
        
        results_df = df_input.copy()
        
        # Ensure columns exist
        if 'predicted_emotion' not in results_df.columns:
            results_df['predicted_emotion'] = None
        if 'predicted_topic' not in results_df.columns:
            results_df['predicted_topic'] = None
            
        # Update with existing data where indices match
        results_df.update(df_existing)
    else:
        results_df = df_input.copy()
        results_df['predicted_emotion'] = None
        results_df['predicted_topic'] = None

    model = setup_model()

    # 3. Iterate
    save_counter = 0
    
    for index, row in tqdm(results_df.iterrows(), total=len(results_df)):
        # Skip if already processed
        if pd.notna(row['predicted_emotion']):
            continue

        title = row['post_title']
        body = row['comment_body']
        
        if pd.isna(body) or str(body).strip() == "":
            results_df.at[index, 'predicted_emotion'] = "Empty"
            results_df.at[index, 'predicted_topic'] = "Other, Irrelevant, Noise"
            continue

        emotion, topic = analyze_comment(model, title, body)
        
        results_df.at[index, 'predicted_emotion'] = emotion
        results_df.at[index, 'predicted_topic'] = topic

        save_counter += 1
        time.sleep(0.1)
        
        if save_counter >= 10:
            results_df.to_csv(OUTPUT_FILE, index=False)
            save_counter = 0

    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
