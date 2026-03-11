import requests
import pandas as pd
import time

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def smart_request(url):
    """Handles the 429 block by waiting and retrying."""
    for i in range(5):
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            wait = (i + 1) * 20 
            print(f"⚠️ Blocked (429). Waiting {wait}s...")
            time.sleep(wait)
        else:
            print(f"Error {response.status_code}")
            return None
    return None

def fetch_posts(subreddit, limit=100):
    posts = []
    after = None
    min_comments = 10
    
    print(f"\n--- Searching r/{subreddit} for posts with >={min_comments} comments ---")
    while len(posts) < limit:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=50"
        if after: url += f"&after={after}"
            
        data = smart_request(url)
        if not data: break
                
        current_batch = data['data']['children']
        for post in current_batch:
            p_data = post['data']
            if p_data.get('num_comments', 0) >= min_comments:
                posts.append({
                    'post_id': p_data['id'],
                    'subreddit': subreddit,
                    'title': p_data['title'],
                    'permalink': p_data['permalink']
                })
            if len(posts) >= limit: break
            
        after = data['data']['after']
        if not after: break
        time.sleep(2)
            
    print(f"Fetched {len(posts)} qualifying posts.")
    return posts

def fetch_comments_for_post(post_data, limit=10):
    url = f"https://www.reddit.com{post_data['permalink']}.json?sort=top"
    comments_list = []
    
    data = smart_request(url)
    if data and len(data) > 1:
        children = data[1]['data']['children']
        for comment in children[:limit]:
            c_data = comment['data']
            if 'body' in c_data:
                comments_list.append({
                    'subreddit': post_data['subreddit'],
                    'post_title': post_data['title'],
                    'comment_body': c_data['body']
                })
    return comments_list

if __name__ == "__main__":
    all_comments = []
    
    # Process Liberal and Conservative subreddits
    for sub in ["Liberal", "Conservative"]:
        posts = fetch_posts(sub, limit=200)
        for i, post in enumerate(posts):
            if i % 10 == 0: print(f"Scraping comments for {sub} post {i}...")
            all_comments.extend(fetch_comments_for_post(post, limit=5))
            time.sleep(1.5) # Be polite to avoid 429s
    
    # Save to a single file
    df = pd.DataFrame(all_comments)
    df.to_csv("new_raw_reddit_comments.csv", index=False)
    print(f"\nSUCCESS: Scraped {len(df)} comments. Saved to 'raw_reddit_comments.csv'.")