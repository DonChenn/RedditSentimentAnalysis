import kagglehub
from kagglehub import KaggleDatasetAdapter

# Updated handle and correct filename
dataset_handle = "fatimamuhammadhaneef/1-million-reddit-comments-from-40-subreddits"
file_path = "kaggle_RC_2019-05.csv" 

try:
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset_handle,
        file_path
    )
    print("Success! Data loaded.")
    subreddit_counts = df['subreddit'].value_counts()

    print("Subreddit comment distribution:")
    print(subreddit_counts)
except Exception as e:
    print(f"Still hitting an error: {e}")