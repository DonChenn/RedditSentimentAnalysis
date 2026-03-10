import json
import time
from collections import Counter, defaultdict

import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from RedditAnalysisApplication import (
    TOPICS, SENTIMENT_MAP,
    load_device, load_sentiment_model, load_political_model, load_topic_model,
    predict_sentiments, predict_political, predict_topics,
    fetch_post_batch, fetch_top_comments,
)

app = Flask(__name__)
CORS(app)

# load models on start
print("Loading models...")
_device = load_device()
_sent_tok, _sent_model, _thresholds = load_sentiment_model(_device)
_pol_tok,  _pol_model               = load_political_model(_device)
_vectorizer, _nmf_model             = load_topic_model()
print(f"All models ready on {_device}\n")


# helpers
def _emotion_summary(pol, pol_counts, emotion_by_pol, bucket_by_pol, topic_names):
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


def _sse(event_type: str, payload: dict):
    data = json.dumps({"type": event_type, **payload})
    return f"data: {data}\n\n"


def _run_analysis_stream(topic_ids: list[int], target_posts: int = 10,
                         max_pages: int = 20, comments_per_post: int = 10):
    """Generator: yields SSE-formatted strings as analysis progresses."""

    topic_names = " | ".join(TOPICS[t] for t in topic_ids)

    yield _sse("status", {"message": f"Scraping r/politics for topic: {topic_names}"})

    matched_posts = []
    all_comments  = []
    after         = None
    pages_scraped = 0
    posts_seen    = set()

    # scrape
    while len(matched_posts) < target_posts and pages_scraped < max_pages:
        posts, after = fetch_post_batch(after, sort="hot")
        if not posts:
            yield _sse("status", {"message": "No more posts returned."})
            break
        pages_scraped += 1

        new_posts = [p for p in posts if p['post_id'] not in posts_seen]
        posts_seen.update(p['post_id'] for p in new_posts)
        if not new_posts:
            if not after:
                break
            continue

        texts = [p['title'] + ' ' + p['selftext'] for p in new_posts]
        topic_ids_pred, topic_scores = predict_topics(texts, _vectorizer, _nmf_model)

        for post, tid, tscore in zip(new_posts, topic_ids_pred, topic_scores):
            if tid in topic_ids:
                matched_posts.append({**post, 'topic_id': tid, 'topic_score': tscore})
                yield _sse("match", {
                    "count":  len(matched_posts),
                    "target": target_posts,
                    "title":  post['title'],
                    "topic":  TOPICS[tid],
                })

        yield _sse("status", {
            "message": f"Page {pages_scraped}: {len(matched_posts)}/{target_posts} posts matched"
        })

        if not after:
            break
        time.sleep(1.5)

    if not matched_posts:
        yield _sse("error", {"message": "No posts matched the target topic."})
        return

    # fetch comments
    yield _sse("status", {"message": f"Fetching comments from {len(matched_posts)} posts..."})
    for i, post in enumerate(matched_posts):
        comments = fetch_top_comments(post['permalink'], limit=comments_per_post)
        for c in comments:
            all_comments.append({'post_title': post['title'], 'text': c})
        yield _sse("status", {
            "message": f"Comments [{i+1}/{len(matched_posts)}]: {post['title'][:55]}"
        })
        time.sleep(1.0)

    if not all_comments:
        yield _sse("error", {"message": "No comments collected."})
        return

    # run models 
    texts = [c['text'] for c in all_comments]
    yield _sse("status", {"message": f"Running sentiment model on {len(texts)} comments..."})
    sentiments = predict_sentiments(texts, _sent_tok, _sent_model, _thresholds, _device)

    yield _sse("status", {"message": "Running political bias model..."})
    political, pol_confidence = predict_political(texts, _pol_tok, _pol_model, _device)

    for i, c in enumerate(all_comments):
        c['sentiment']      = sentiments[i]
        c['political']      = political[i]
        c['pol_confidence'] = pol_confidence[i]

    pol_counts     = Counter(c['political'] for c in all_comments)
    total          = len(all_comments)
    lib_pct        = pol_counts.get('Liberal', 0) / total
    con_pct        = pol_counts.get('Conservative', 0) / total
    dominant_pol   = 'Liberal' if lib_pct >= con_pct else 'Conservative'
    dominant_pct   = max(lib_pct, con_pct)
    avg_confidence = float(np.mean(pol_confidence))

    emotion_by_pol: dict[str, Counter] = defaultdict(Counter)
    bucket_by_pol:  dict[str, Counter] = defaultdict(Counter)

    for c in all_comments:
        pol = c['political']
        for emotion in c['sentiment']:
            emotion_by_pol[pol][emotion] += 1
        pos = sum(1 for e in c['sentiment'] if e in SENTIMENT_MAP['positive'])
        neg = sum(1 for e in c['sentiment'] if e in SENTIMENT_MAP['negative'])
        bucket = 'positive' if pos > neg else ('negative' if neg > pos else 'neutral')
        bucket_by_pol[pol][bucket] += 1

    lib_summary = _emotion_summary(
        'Liberal', pol_counts, emotion_by_pol, bucket_by_pol, topic_names)
    con_summary = _emotion_summary(
        'Conservative', pol_counts, emotion_by_pol, bucket_by_pol, topic_names)

    result = {
        "topic_names":     topic_names,
        "posts_found":     len(matched_posts),
        "total_comments":  total,
        "dominant_pol":    dominant_pol,
        "dominant_pct":    round(dominant_pct, 4),
        "avg_confidence":  round(avg_confidence, 4),
        "political": {
            "Liberal":      pol_counts.get('Liberal', 0),
            "Conservative": pol_counts.get('Conservative', 0),
            "lib_pct":      round(lib_pct, 4),
            "con_pct":      round(con_pct, 4),
        },
        "emotions": {
            pol: {
                "top5":    dict(emotion_by_pol[pol].most_common(5)),
                "buckets": dict(bucket_by_pol[pol]),
            }
            for pol in ['Liberal', 'Conservative']
        },
        "summaries": {
            "Liberal":      lib_summary,
            "Conservative": con_summary,
        },
    }

    yield _sse("result", {"data": result})


# Routes

@app.get("/api/topics")
def get_topics():
    return jsonify([{"id": tid, "name": name} for tid, name in TOPICS.items()])


@app.get("/api/analyze")
def analyze():
    topic_id_str = request.args.get("topic_id", "")
    if not topic_id_str.isdigit():
        return jsonify({"error": "topic_id must be a number"}), 400
    topic_id = int(topic_id_str)
    if topic_id not in TOPICS:
        return jsonify({"error": f"topic_id {topic_id} not found"}), 400

    def stream():
        try:
            yield from _run_analysis_stream(
                topic_ids      = [topic_id],
                target_posts   = 10,
                max_pages      = 20,
                comments_per_post = 10,
            )
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    app.run(port=8000, debug=False)
