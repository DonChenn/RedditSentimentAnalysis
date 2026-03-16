External Libraries:
    pandas (https://pypi.org/project/pandas/)
    numpy (https://pypi.org/project/numpy/)
    scikit-learn (https://pypi.org/project/scikit-learn/)
    requests (https://pypi.org/project/requests/)
    nltk (https://pypi.org/project/nltk/)
    vaderSentiment (https://pypi.org/project/vaderSentiment/)
    torch (https://pypi.org/project/torch/)
    torchvision (https://pypi.org/project/torchvision/)
    torchaudio (https://pypi.org/project/torchaudio/)
    transformers (https://pypi.org/project/transformers/)
    accelerate>=1.1.0 (https://pypi.org/project/accelerate/)
    datasets (https://pypi.org/project/datasets/)
    bertopic (https://pypi.org/project/bertopic/)
    gensim (https://pypi.org/project/gensim/)
    matplotlib (https://pypi.org/project/matplotlib/)
    seaborn (https://pypi.org/project/seaborn/)
    tqdm (https://pypi.org/project/tqdm/)
    kagglehub (https://pypi.org/project/kagglehub/)
    google-genai>=0.1.0 (https://pypi.org/project/google-genai/)
    dotenv (https://pypi.org/project/python-dotenv/) 
    UMAP (https://umap-learn.readthedocs.io/en/latest/api.html)

Publicly available code:

    VADER sentiment library (vaderSentiment). Used for baseline inference                      
    BERTopic library. (https://maartengr.github.io/BERTopic/algorithm/algorithm.html): Used for baseline inference 
    NMF (Non-Negative Matrix Factorization) library (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html): 
    Used for baseline inference
    UMAP (https://umap-learn.readthedocs.io/en/latest/api.html): Used for baseline inference
    bucketresearch/politicalBiasBERT (https://huggingface.co/bucketresearch/politicalBiasBERT):
    Used for zero-shot baseline inference and as base model for fine-tuned political classifier

Code written by team:

    Topic Classifier Model/k_means_bertopic_seeded.py
        BERTopic model that Uses K-means clustering and topic seeding trained on dataset to classify topics(was done on google colab but ipynb file did 
        not want to upload to github so uploaded python file instead) (227 lines)

    Topic Classifier Model/TFIDFBaselineTopicModel.py
        TF-IDF model that uses NNMF trained on a dataset to classify topics (116 lines)

    Topic Classifier Model/stopwords.py
        Custom stop list filled with general stop words in political comments (116 lines)

    Topic Classifier Model/combinedtrainingdataset.py
        Pipeline script that combines Liberal vs Conservative dataset (title + body text) and 1 million reddit comments (25,000 r/politics subreddit)
        (44 lines)

    Sentiment model training: 
        GoEmotionsRoBERTa.ipynb: RoBERTa model trained on learning rate of 5e-5, 4 epochs, no warmup, linear class weights, weighted f1 (448 lines)
        GoEmotionsImprovedRoBERTa.ipynb: RoBERTa model trained on learning rate of 3e-5, 6 epochs, 0.1 warmup, sqrt class weights, macro f1 (171 lines)
        GoEmotionsCNN.ipynb: CNN sentiment model trained on GoEmotions dataset (425 lines)
        GoEmotionsVADER.ipynb: VADER baseline model (154 lines)
        GoEmotionsFewShot.ipynb: Gemini Few Shot baseline model (243 lines)

    Political model training:
        politicalBiasBert.py: Zero Shot politicalBiasBERT baseline with neutral class filtering and probability renormalization for binary evaluation (160 lines)
        politicalBiasBertfinetunedCODE.py: Fine Tuned politicalBiasBERT trained on Liberals vs Conservatives dataset with 2 class classification head (256 lines)
        politicalBiasBert_finetuned_hyperparametertuned_code.py: Fine Tuned + Hyperparameter Tuned politicalBiasBERT with extensive tuning across learning rate, epochs, dropout, sequence length, batch size, weight decay, 	gradient clipping, warmup steps, and lr scheduler (257 lines)
        tfidf.py: TF IDF + Logistic Regression political baseline (154 lines)

    TestDatasetScraper.py: Reddit API scraper that collects comments from political subreddits for final evaluation. (~86 lines)

    LLM_labeling_improved.py: Gemini 2.0 Flash labeling pipeline that assigns GoEmotions and topic labels to scraped Reddit comments to construct the test dataset. (~462 lines)

    project.ipynb: Pipeline for all three advanced models on our test evaluation dataset (585 lines)

    baselinesCombined.py: Pipeline for all three baseline models on our test evaluation dataset (270 lines)
