import sys
sys.path.append('./src')

from config import *
from data_preprocessing import load_and_clean_data
from feature_engineering import FeatureBuilder
from model_training import train_xgb, evaluate

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Load and preprocess data
df = load_and_clean_data(DATA_PATH)
y = df['fraudulent']

# Split data
train_df, test_df, y_train, y_test = train_test_split(df, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)

# Feature engineering
builder = FeatureBuilder(TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE)
X_train = builder.fit_transform(train_df)
X_test = builder.transform(test_df)

# SMOTE balancing
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train & Evaluate
model = train_xgb(X_train_balanced, y_train_balanced)
f1 = evaluate(model, X_test, y_test)

# Save model + vectorizer + encoder for future prediction
joblib.dump(model, 'models/final_model.pkl')
joblib.dump(builder.tfidf, 'models/vectorizer.pkl')
joblib.dump(builder.ohe, 'models/ohe.pkl')

print("Training complete. Model and vectorizer saved.")
