External Libraries:
    pandas (https://pandas.pydata.org/)
    scikit-learn (https://scikit-learn.org/stable/)
    bertopic (https://maartengr.github.io/BERTopic/index.html)
    umap (https://pypi.org/project/umap-learn/)
    os (https://docs.python.org/3/library/os.html)
    sys (https://docs.python.org/3/library/sys.html)
    pickle (https://docs.python.org/3/library/pickle.html)
    kagglehub (https://github.com/Kaggle/kagglehub)





Publicly available code:





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

