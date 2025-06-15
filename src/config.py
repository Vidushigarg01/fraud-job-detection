import os
RANDOM_SEED = 42
TEST_SIZE = 0.20
TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'fake_job_postings.csv')

