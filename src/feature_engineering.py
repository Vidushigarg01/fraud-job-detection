# src/feature_engineering.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import numpy as np

class FeatureBuilder:
    def __init__(self, tfidf_max_features, tfidf_ngram_range):
        self.text_fields = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        self.categorical_fields = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
        self.binary_fields = ['telecommuting', 'has_company_logo', 'has_questions']
        self.tfidf = TfidfVectorizer(stop_words='english',
                                     ngram_range=tfidf_ngram_range,
                                     max_features=tfidf_max_features)
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    def fit_transform(self, df):
        X_text = df[self.text_fields].agg(' '.join, axis=1)
        X_text_tfidf = self.tfidf.fit_transform(X_text)
        X_struct = df[self.binary_fields + self.categorical_fields]
        X_struct_ohe = self.ohe.fit_transform(X_struct)
        X_final = hstack([X_text_tfidf, X_struct_ohe])
        return X_final

    def transform(self, df):
        X_text = df[self.text_fields].agg(' '.join, axis=1)
        X_text_tfidf = self.tfidf.transform(X_text)
        X_struct = df[self.binary_fields + self.categorical_fields]
        X_struct_ohe = self.ohe.transform(X_struct)
        X_final = hstack([X_text_tfidf, X_struct_ohe])
        return X_final

    def get_vectorizer(self):
        return self.tfidf

    
