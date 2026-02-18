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

INPUT_FILE = 'raw_reddit_comments.csv'
OUTPUT_FILE = 'labeled_reddit_comments.csv'

# Use the stable 2.0 Flash model
MODEL_NAME = 'gemini-2.0-flash'

EMOTIONS = [
    "admiration", "amusement", "approval", "caring", "desire", "excitement", 
    "gratitude", "joy", "love", "optimism", "pride", "relief", "anger", 
    "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", 
    "fear", "grief", "nervousness", "remorse", "sadness", "neutral", 
    "realization", "surprise", "curiosity", "confusion"
]

TOPICS_LIST = [
    "Other / Irrelevant / Noise",
    "ukraine, russia, russian, invasion",
    "female, sex, gender, womens",
    "poll, approval, signed, infrastructure",
    "vaccine, covid, covid19, pandemic",
    "capitalism, value, labor, capital",
    "canadian, canada, truckers, bank",
    "race, racism, black, critical",
    "florida, gov, poll, governor",
    "sanctions, russia, ukraine, russian",
    "land, housing, property, rent",
    "notice, pretty, trust, difference",
    "anarchocapitalist, anarchist, property, anarchists",
    "cuba, parents, code, vaccine",
    "fox, news, fox news, media",
    "communist, communism, communists, film",
    "strike, workers, strikes, labor",
    "lgbtq, trans, kids, texas",
    "probe, campaign, white house, investigation",
    "climate, climate change, change, water",
    "inflation, fed, reserve, prices",
    "libertarian, libertarians, folks, freedom",
    "china, chinese, chinas, africa",
    "lenin, deaths, ussr, soviet",
    "minimum wage, minimum, wage, wages",
    "mandates, mandate, super, face",
    "bitcoin, peter, regulation, regulate",
    "student, debt, loan, loans",
    "afghanistan, mistakes, funds, reserves",
    "social democracy, social, democracy, democratic",
    "chance, leftist, leftists, average",
    "socialism, soviet, soviet union, socialists",
    "unions, union, collective, employers",
    "capitol, arrested, charged, jan",
    "store, social, social media, site",
    "taxes, income, net, nov",
    "dsa, east, peace, humanity",
    "texas, border, primary, district",
    "county, fraud, officials, ballot",
    "books, book, german, advance",
    "tax, billionaires, taxes, gains",
    "putin, weak, emboldened, playing",
    "map, alabama, congressional, district",
    "prison, sexual, assault, police",
    "supreme, supreme court, court, pick",
    "womens, assault, south, faces",
    "fascism, fascist, demand, nazis",
    "suppress, anymore, agenda, votes",
    "organizing, workers, union, store",
    "fox news, fox, targeted, news",
    "schools, education, school, parents",
    "police, cops, authority, security",
    "child, childcare, credit, families",
    "pelosi, infrastructure, senate, resolution",
    "poll, ukraine, putin, majority",
    "democracy, australian, opposing, direct",
    "rights, senators, act, blocking",
    "run, running, hes, coming",
    "healthcare, insurance, doctor, universal",
    "putin, putins, genius, praise",
    "amazon, employees, value, customers",
    "australian, australia, immigration, racist",
    "china, chinese, activities, nato",
    "leftist, platforms, channel, informed",
    "labour, uk, pm, 11",
    "abortion, access, baby, laws",
    "speech, trade, business, sector",
    "dsa, city council, council, eye",
    "gun, guns, violence, mass",
    "georgia, voters, ballot, restrictions",
    "nato, finland, expansion, sweden",
    "energy, oil, oil gas, fuel",
    "violent, farright, domestic, terror",
    "ancap, anarchist, property, personal",
    "rally, documentary, campaign, donations",
    "coal, trillion, gas, backs",
    "imperialism, global, plenty, south",
    "wealth, rich, worlds, poor",
    "military, defense, firms, foreign",
    "backed, dem, interview, defeated",
    "war, wars, class, warfare",
    "oil, africa, official, france",
    "presidential, leftist, dem, elections",
    "educated, goals, anti, work",
    "freedom, speech, culture, rights",
    "peoples, army, communist, tear",
    "abortion, passes, senate, court",
    "probe, leading, investigation, criminal",
    "gas, germany, sanctions, prices",
    "join, currently, group, members",
    "marx, science, church, marxism",
    "truly, line, went,",
    "social media, content, internet, intelligence",
    "public, welfare, rid, policies",
    "green, capitalist, guard, presidential",
    "india, communist, killed, communists",
    "mark, 2022, chapter, 1st",
    "matters, coming, midterms, nov",
    "jan, resolution, panel, committee",
    "texas, abortion, law, site",
    "stolen, running, blasts, office",
    "easy, wake, walk, steps",
    "florida, seat, special, city council",
    "progressive, advocates, winning, mainstream",
    "voters, congressional, district, campaign",
    "bomb, slams, knows, shocking",
    "cpac, poll, wins, presidential",
    "48, questions, ,",
    "senate, pass, rights, end",
    "organizations, worker, largest, shares",
    "drugs, price, insurance, health",
    "rep, responsibility, eric, analysis",
    "striking, contract, letter, rejected",
    "demands, actions, class, solidarity",
    "questions, email, activists, earn",
    "books, banning, burning, book",
    "sports, womens, girls, trans",
    "nationalist, conference, rep, white",
    "equality, financial, facts, crisis",
    "revolution, revolutionary, documentary, violent",
    "banned, speech, hate, surprising",
    "california, term, housing, san",
    "leftwing, independence, rightwing, oppressed",
    "russian, propaganda, x200b, russia",
    "internet, cities, public, cars",
    "sanctions, 2018, opposition, iraq",
    "progressives, progressive, candidates, leaning",
    "pelosi, congress, stock, ban",
    "countries, denmark, finland, norway",
    "market, anarchist, resources, planned",
    "online, popularity, minutes, deep",
    "eu, european, europe, federation",
    "expansion, buy, health, proves",
    "lawyers, plan, officials, green",
    "accountable, shut, activists, children",
    "unemployment, benefits, pandemic, cutting",
    "black, lives matter, criminal, lie",
    "stephen, supreme court, supreme, justice",
    "happened, happen, stopping, fault",
    "cpac, 2022, crowd, friday",
    "social media, media, account, organizations",
    "russians, protesting, streets, invasion",
    "lgbt, videos, pride, straight",
    "blm, protests, protest, bullets",
    "sue, media, combat, plans",
    "context, soviet union, red, books",
    "white, way, ,",
    "dropped, number, lost, pandemic",
    "paul, fauci, hopes, misinformation",
    "immigration, immigrants, germany, economy",
    "debt, trillion, 30, decades",
    "economically, americans, appear, crowd",
    "courts, justice, victim, private",
    "solidarity, stand, forever, farmers",
    "2022, wednesday, standing, panel",
    "fear, jail, realized, learned",
    "committee, jan, issued, eric",
    "libertarian, wars, conflict, ukraine",
    "monopoly, competition, market, firm",
    "blm, lives matter, white, rallies",
    "coalition, germany, paper, mentioned",
    "directly, save, skeptical, peter",
    "supreme court, viable, supreme, npr",
    "market, black, incentives, range",
    "abortion, womens, ban, protest",
    "coup, michael, conference, attempt",
    "grand, probe, speaker, sex",
    "presidents, stock, worldwide, revolutions",
    "north, korea, north korea, nuclear",
    "aid, fell, poverty, boost",
    "japan, imperialist, nuclear, violating",
    "nationalists, socialists, lives matter, island",
    "tax, income tax, rich, cutting",
    "hitler, fan, burn, agrees",
    "jan, committee, records, documents",
    "seize, cnn, committee, attorney general",
    "food, mutual, aid, count",
    "fraud, ny, significant, investigation",
    "york, sexual, governor, new york",
    "ballot, midterm, advantage, congressional",
    "intelligence, fbi, journalists, forum",
    "dies, senate, letter, majority",
    "nationalism, proud, lie, code",
    "peace, secure, reason, norway",
    "controls, economics, greatest, paul",
    "land, afford, rent, lower",
    "marxist, classes, discourse, series",
    "brown, supreme court, supreme, court",
    "attempted, blm, murder, charged",
    "requirements, germany, nato, proves",
    "poverty, absolute, developed, poor",
    "organization, new york, york, midterm",
    "announce, ted, house, highly",
    "victims, assault, crime, victim",
    "quote, add, 75, suggested",
    "marriage, child, legally, girls"
]

def setup_model():
    genai.configure(api_key=API_KEY)
    
    # 1. Config: Lower temperature for consistency, JSON mime type
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "max_output_tokens": 1024,
        "response_mime_type": "application/json",
    }
    
    # 2. Safety: Disable blocks so political content doesn't return empty
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    formatted_topics = "\n".join([f"Index {i}: {t}" for i, t in enumerate(TOPICS_LIST)])

    system_instruction = (
        "You are an expert data annotator. Your task is to analyze Reddit comments "
        "for sentiment and political topic.\n"
        f"1. EMOTION: Select exactly one emotion from: {', '.join(EMOTIONS)}. "
        "If unsure, choose 'neutral'.\n"
        f"2. TOPIC: Select exactly one Topic Index (integer) from the following list:\n{formatted_topics}\n"
        "If no topic matches perfectly, use Index 0 (Other/Noise).\n"
        "Return a JSON object with keys 'emotion' and 'topic_index'."
    )
    
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=system_instruction
    )
    return model

def analyze_comment(model, title, body):
    # Add unique ID to prompt to prevent caching bugs
    request_id = str(uuid.uuid4())
    prompt = f"RequestID: {request_id}\nPost Title: {title}\nComment Body: {body}"
    
    # Retry loop for 429s or empty responses
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            
            # Check for safety blocks or empty returns
            if not response.parts:
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    return "blocked", 0
                if attempt < 2:
                    time.sleep(2)
                    continue
                return "neutral", 0
            
            # Robust JSON cleaning
            text_content = response.text.strip()
            # Remove Markdown code blocks if present
            if "```" in text_content:
                text_content = re.sub(r'```json\s*|\s*```', '', text_content)
            
            result = json.loads(text_content)
            
            # Robust parsing of Emotion
            emotion = result.get("emotion", "neutral")
            if not emotion: emotion = "neutral"
            
            # Robust parsing of Topic Index (Handle string vs int)
            topic_raw = result.get("topic_index", 0)
            try:
                topic_idx = int(topic_raw)
            except (ValueError, TypeError):
                topic_idx = 0
                
            # Bounds check
            if not (0 <= topic_idx < len(TOPICS_LIST)):
                topic_idx = 0
                
            return emotion, topic_idx

        except Exception as e:
            print(f"Error on attempt {attempt}: {e}")
            time.sleep(2) # Backoff before retry
            
    return "ERROR", -1

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. Load Data
    df_input = pd.read_csv(INPUT_FILE)
    print(f"Input file has {len(df_input)} rows.")

    # 2. Resume Logic
    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming from {OUTPUT_FILE}...")
        df_existing = pd.read_csv(OUTPUT_FILE)
        
        # Merge existing labels into the main dataframe
        results_df = df_input.copy()
        
        # Initialize columns if they don't exist
        if 'predicted_emotion' not in results_df.columns:
            results_df['predicted_emotion'] = None
        if 'predicted_topic' not in results_df.columns:
            results_df['predicted_topic'] = None
            
        # Update rows that have already been labeled in OUTPUT_FILE
        # We assume the index alignment is preserved. 
        # A safer way is to update based on index if the files are identical in length.
        results_df.update(df_existing)
    else:
        results_df = df_input.copy()
        results_df['predicted_emotion'] = None
        results_df['predicted_topic'] = None

    model = setup_model()

    save_counter = 0
    
    # 3. Processing Loop
    for index, row in tqdm(results_df.iterrows(), total=len(results_df)):
        # Skip if already processed
        current_emotion = row['predicted_emotion']
        if pd.notna(current_emotion) and current_emotion != "ERROR" and current_emotion != "":
            continue

        title = row['post_title']
        body = row['comment_body']
        
        # Handle empty comments
        if pd.isna(body) or str(body).strip() == "":
            results_df.at[index, 'predicted_emotion'] = "neutral"
            results_df.at[index, 'predicted_topic'] = 0
            continue

        emotion, topic_idx = analyze_comment(model, title, body)
        
        results_df.at[index, 'predicted_emotion'] = emotion
        results_df.at[index, 'predicted_topic'] = topic_idx

        save_counter += 1
        
        # Save every 20 rows
        if save_counter >= 10:
            results_df.to_csv(OUTPUT_FILE, index=False)
            save_counter = 0
            
        # Rate limit protection
        time.sleep(1.0)

    # Final Save
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
