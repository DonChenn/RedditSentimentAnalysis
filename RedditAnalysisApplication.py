import json
import pickle
import time
from collections import Counter, defaultdict

import numpy as np
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (compatible; RedditAnalysisApp/1.0; '
        '+https://github.com/research-project)'
    )
}

SENTIMENT_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

SENTIMENT_MAP = {
    "positive": {
        'admiration', 'amusement', 'approval', 'caring', 'desire',
        'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'
    },
    "negative": {
        'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
        'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'
    },
    "neutral": {'neutral', 'realization', 'surprise', 'curiosity', 'confusion'},
}

TOPICS = {
    0: "Noise/Filler", 
    1: "Mueller Investigation", 
    2: "URLs & Metadata",
    3: "Bill Barr Perjury", 
    4: "Life & Big Questions", 
    5: "Senate Hearings",
    6: "Mueller Report Release", 
    7: "Russia-Ukraine & NATO", 
    8: "Special Counsel",
    9: "House Oversight", 
    10: "Accountability", 
    11: "Media Bias",
    12: "Congress vs Law", 
    13: "Political Talking Points", 
    14: "How Politics Works",
    15: "Capitalism vs Socialism", 
    16: "Reddit Community", 
    17: "NY Politics",
    18: "Senate & McConnell", 
    19: "Identity Politics", 
    20: "Leaked Documents",
    21: "Arguing Definitions", 
    22: "Socialism & Welfare", 
    23: "War & Anti-War",
    24: "DOJ Charges", 
    25: "Impeachment", 
    26: "Health Care",
    27: "Campaign Finance", 
    28: "Elections & Voter Base", 
    29: "Jobs & Fair Pay",
    30: "Power & Influences",
    31: "Russia Sanctions", 
    32: "Political Skepticism",
    33: "Strong Public Reactions", 
    34: "News Coverage Critique", 
    35: "Supreme Court",
    36: "Barr's Summary", 
    37: "Race & Civil Rights", 
    38: "Approval Ratings",
    39: "Progressive Sentiment", 
    40: "Attorney General Rating", 
    41: "Taxes & Wealthy",
    42: "Violence & Extremism", 
    43: "Law & Abortion", 
    44: "Calls for Action",
    45: "Legal Rights", 
    46: "Unions & Strikes", 
    47: "Obstruction",
    48: "Lying Under Oath", 
    49: "China & Communism",
}

POLITICAL_LABEL_MAP = {0: 'Liberal', 1: 'Conservative'}

MODEL_DIR = "./.saved_models"


# LOAD MODELS
def load_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_sentiment_model(device):
    path = f"{MODEL_DIR}/FinalSentimentModel"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, problem_type="multi_label_classification"
    ).to(device).eval()
    with open(f"{path}/optimal_thresholds.json") as f:
        thresh_dict = json.load(f)
    thresholds = np.array([thresh_dict[name] for name in SENTIMENT_LABELS])
    print(f"  sentiment model loaded ({path})")
    return tokenizer, model, thresholds


def load_political_model(device):
    model_name = "bucketresearch/politicalBiasBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    weights_path = f"{MODEL_DIR}/politicalBiasBERT_finetuned_best.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device).eval()
    print(f"  political model loaded ({weights_path})")
    return tokenizer, model


def load_topic_model():
    with open(f"{MODEL_DIR}/tfidf_topic_model.pkl", 'rb') as f:
        data = pickle.load(f)
    print(f"  topic model loaded ({data['nmf_model'].n_components} topics)")
    return data['vectorizer'], data['nmf_model']


# PREDICT MODELS
def predict_sentiments(texts, tokenizer, model, thresholds, device, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.sigmoid(model(**inputs).logits).cpu().numpy()
        for prob in probs:
            preds = [SENTIMENT_LABELS[j] for j, t in enumerate(thresholds)
                     if prob[j] >= t]
            results.append(preds if preds else ['neutral'])
    return results


def predict_political(texts, tokenizer, model, device, batch_size=16):
    results = []
    confidences = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
        for pred, prob in zip(preds, probs):
            results.append(POLITICAL_LABEL_MAP[pred])
            confidences.append(float(prob.max()))
    return results, confidences


def predict_topics(texts, vectorizer, nmf_model):
    tfidf = vectorizer.transform(texts)
    dist = nmf_model.transform(tfidf)
    return dist.argmax(axis=1).tolist(), dist.max(axis=1).tolist()


# FINDING TOPIC
def resolve_topic_id(query: str) -> list[int]:
    """
    Accepts either a numeric topic ID or a keyword string.
    Returns a list of matching topic IDs.
    """
    if query.isdigit():
        tid = int(query)
        if tid in TOPICS:
            return [tid]
        raise ValueError(f"Topic ID {tid} not in range 0-49")

    query_lower = query.lower()
    matches = [tid for tid, name in TOPICS.items()
               if query_lower in name.lower()]
    if not matches:
        raise ValueError(
            f"No topic found matching '{query}'.\n"
            f"Available topics:\n" +
            "\n".join(f"  {tid:2d}: {name}" for tid, name in TOPICS.items())
        )
    return matches


# SCRAPING
def smart_request(url: str, retries: int = 5):
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = (attempt + 1) * 20
                print(f"  [429] rate-limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [HTTP {resp.status_code}] {url}")
                return None
        except requests.RequestException as e:
            print(f"  [error] {e}")
            time.sleep(5)
    return None


def fetch_post_batch(after: str | None, subreddit: str = "politics",
                     sort: str = "hot", limit: int = 50):
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
    if after:
        url += f"&after={after}"
    data = smart_request(url)
    if not data:
        return [], None
    children = data['data']['children']
    posts = []
    for child in children:
        p = child['data']
        posts.append({
            'post_id':   p['id'],
            'title':     p.get('title', ''),
            'selftext':  p.get('selftext', ''),
            'permalink': p.get('permalink', ''),
            'score':     p.get('score', 0),
            'num_comments': p.get('num_comments', 0),
        })
    return posts, data['data'].get('after')


def fetch_top_comments(permalink: str, limit: int = 10):
    url = f"https://www.reddit.com{permalink}.json?sort=top&limit={limit}"
    data = smart_request(url)
    comments = []
    if data and len(data) > 1:
        for child in data[1]['data']['children']:
            c = child['data']
            if 'body' in c and c['body'] not in ('[removed]', '[deleted]'):
                comments.append(c['body'])
    return comments


# RUN ANALYSIS

def run_analysis(
    target_topic_ids: list[int],
    target_posts: int = 10,
    max_scrape_pages: int = 20,
    comments_per_post: int = 10,
    sort: str = "hot",
):
    device = load_device()
    print(f"\nDevice: {device}")
    print("Loading models...")
    sent_tok, sent_model, thresholds = load_sentiment_model(device)
    pol_tok,  pol_model              = load_political_model(device)
    vectorizer, nmf_model            = load_topic_model()
    print()

    topic_names = " | ".join(TOPICS[t] for t in target_topic_ids)
    print(f"Target topic(s): {topic_names}  (IDs: {target_topic_ids})")
    print(f"Goal: {target_posts} matching posts\n")

    matched_posts   = []   # posts whose title matches the topic
    all_comments    = []   # (post_title, comment_body) tuples
    after           = None
    pages_scraped   = 0
    posts_seen      = set()

    # scrape until we have enough topic-matched posts
    while len(matched_posts) < target_posts and pages_scraped < max_scrape_pages:
        posts, after = fetch_post_batch(after, sort=sort)
        if not posts:
            print("  No more posts returned — stopping scrape.")
            break
        pages_scraped += 1

        new_posts = [p for p in posts if p['post_id'] not in posts_seen]
        posts_seen.update(p['post_id'] for p in new_posts)

        if not new_posts:
            if not after:
                break
            continue

        # run topic model on post titles + selftext
        texts = [p['title'] + ' ' + p['selftext'] for p in new_posts]
        topic_ids, topic_scores = predict_topics(texts, vectorizer, nmf_model)

        for post, tid, tscore in zip(new_posts, topic_ids, topic_scores):
            if tid in target_topic_ids:
                matched_posts.append({**post, 'topic_id': tid, 'topic_score': tscore})
                print(f"  [match {len(matched_posts):2d}/{target_posts}] "
                      f"topic={TOPICS[tid]}  score={tscore:.3f}  "
                      f"title={post['title'][:70]}")

        print(f"  page {pages_scraped}: scanned {len(new_posts)} posts, "
              f"{len(matched_posts)} matched so far")

        if not after:
            print("  Reached end of listing.")
            break
        time.sleep(1.5)   # be polite

    if not matched_posts:
        print("\nNo posts matched the target topic. Try a different topic or increase --max-pages.")
        return

    print(f"\nFetching comments from {len(matched_posts)} matched posts...")
    for i, post in enumerate(matched_posts):
        comments = fetch_top_comments(post['permalink'], limit=comments_per_post)
        for c in comments:
            all_comments.append({'post_title': post['title'], 'text': c})
        print(f"  [{i+1}/{len(matched_posts)}] {len(comments)} comments — {post['title'][:60]}")
        time.sleep(1.0)

    if not all_comments:
        print("\nNo comments collected.")
        return

    # --- run models on comments ---
    texts = [c['text'] for c in all_comments]
    print(f"\nRunning models on {len(texts)} comments...")

    sentiments,  _              = zip(*[None]*2) if not texts else (
        predict_sentiments(texts, sent_tok, sent_model, thresholds, device),
        None
    )
    political, pol_confidence   = predict_political(texts, pol_tok, pol_model, device)
    sentiments                  = predict_sentiments(texts, sent_tok, sent_model, thresholds, device)

    for i, c in enumerate(all_comments):
        c['sentiment']        = sentiments[i]
        c['political']        = political[i]
        c['pol_confidence']   = pol_confidence[i]

    # confidence thresholds
    pol_counts   = Counter(c['political'] for c in all_comments)
    total        = len(all_comments)
    lib_pct      = pol_counts.get('Liberal', 0) / total
    con_pct      = pol_counts.get('Conservative', 0) / total
    dominant_pct = max(lib_pct, con_pct)
    dominant_pol = 'Liberal' if lib_pct >= con_pct else 'Conservative'

    # emotion aggregates per political group
    emotion_by_pol: dict[str, Counter] = defaultdict(Counter)
    bucket_by_pol:  dict[str, Counter] = defaultdict(Counter)

    for c in all_comments:
        pol = c['political']
        for emotion in c['sentiment']:
            emotion_by_pol[pol][emotion] += 1
        # bucket
        pos = sum(1 for e in c['sentiment'] if e in SENTIMENT_MAP['positive'])
        neg = sum(1 for e in c['sentiment'] if e in SENTIMENT_MAP['negative'])
        bucket = 'positive' if pos > neg else ('negative' if neg > pos else 'neutral')
        bucket_by_pol[pol][bucket] += 1

    # NLP SUMMARY

    def emotion_summary(pol: str) -> str:
        """
        Returns a sentence like:
        "Liberals are primarily feeling anger and annoyance about Health Care,
         reflecting a mostly negative overall tone."
        """
        pol_total = pol_counts.get(pol, 0)
        if pol_total == 0:
            return f"{pol}s had no comments in this sample."

        top_emotions = [e for e, _ in emotion_by_pol[pol].most_common(3)]
        buckets      = bucket_by_pol[pol]
        dominant_bucket = max(buckets, key=buckets.get)

        if len(top_emotions) == 1:
            emotions_str = top_emotions[0]
        elif len(top_emotions) == 2:
            emotions_str = f"{top_emotions[0]} and {top_emotions[1]}"
        else:
            emotions_str = f"{top_emotions[0]}, {top_emotions[1]}, and {top_emotions[2]}"

        bucket_phrase = {
            'positive': 'an overall positive tone',
            'negative': 'an overall negative tone',
            'neutral':  'a mostly neutral tone',
        }[dominant_bucket]

        pct = buckets.get(dominant_bucket, 0) / pol_total
        return (
            f"{pol}s are primarily feeling {emotions_str} about {topic_names}, "
            f"reflecting {bucket_phrase} ({pct:.0%} of comments)."
        )

    # STATS
    divider = "=" * 65

    print(f"\n{divider}")
    print("  POLITICAL SENTIMENT ANALYSIS RESULTS")
    print(divider)
    print(f"  Topic(s)    : {topic_names}")
    print(f"  Posts found : {len(matched_posts)}  |  Comments : {total}")
    print(f"  Subreddit   : r/politics")
    print(divider)

    print(f"\n  Political Leaning")
    print(f"  {'Liberal':<14}: {pol_counts.get('Liberal',0):4d}  ({lib_pct:.1%})")
    print(f"  {'Conservative':<14}: {pol_counts.get('Conservative',0):4d}  ({con_pct:.1%})")
    print(f"\n  Dominant leaning : {dominant_pol}  ({dominant_pct:.1%})")

    for pol in ['Liberal', 'Conservative']:
        pol_total = pol_counts.get(pol, 0)
        if pol_total == 0:
            continue
        print(f"\n  [{pol}] — {pol_total} comments")
        buckets = bucket_by_pol[pol]
        for b in ('positive', 'negative', 'neutral'):
            cnt = buckets.get(b, 0)
            print(f"    {b:<10}: {cnt:4d}  ({cnt/pol_total:.1%})")
        top5 = emotion_by_pol[pol].most_common(5)
        print(f"    top emotions : {', '.join(f'{e}({n})' for e, n in top5)}")

    print(f"\n{divider}")
    print("  SUMMARY")
    print(divider)
    lib_summary = emotion_summary('Liberal')
    con_summary = emotion_summary('Conservative')
    print(f"\n  {lib_summary}")
    print(f"\n  {con_summary}")

    avg_confidence = np.mean([c['pol_confidence'] for c in all_comments])
    print(f"\n  Avg political model confidence: {avg_confidence:.3f}")

    print(f"\n{divider}\n")

    return {
        'topic_ids':     target_topic_ids,
        'posts':         matched_posts,
        'comments':      all_comments,
        'political':     dict(pol_counts),
        'dominant':      dominant_pol,
        'dominant_pct':  dominant_pct,
        'emotion_by_pol': {k: dict(v) for k, v in emotion_by_pol.items()},
        'bucket_by_pol':  {k: dict(v) for k, v in bucket_by_pol.items()},
    }


# PROMPT

def prompt_topic() -> list[int]:
    print("\nAvailable topics:")
    for tid, name in TOPICS.items():
        print(f"  {tid:2d}: {name}")

    while True:
        print()
        query = input("Enter a topic name or ID (e.g. 'Healthcare' or '26'): ").strip()
        if not query:
            continue
        try:
            topic_ids = resolve_topic_id(query)
            names = " | ".join(TOPICS[t] for t in topic_ids)
            print(f"  Matched: {names}  (IDs: {topic_ids})")
            return topic_ids
        except ValueError as e:
            print(f"  {e}")


def main():
    print("=" * 65)
    print("  Reddit Political Sentiment Analyzer")
    print("=" * 65)

    topic_ids = prompt_topic()

    run_analysis(
        target_topic_ids  = topic_ids,
        target_posts      = 10,
        max_scrape_pages  = 20,
        comments_per_post = 10,
        sort              = "hot",
    )


if __name__ == "__main__":
    main()
