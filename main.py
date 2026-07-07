import pandas as pd
import re
import nltk
import ssl
import os
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- 1. SSL & NLTK SETUP ---

# Bypassing SSL for certificate errors
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK resources
print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- 2. LOAD DATASET ---
file_path = "IMDB Dataset.csv"

try:
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"✅ Dataset loaded successfully: {len(df)} reviews")
except FileNotFoundError:
    print(f"❌ Error: Could not find '{file_path}'")
    print(f"Current directory: {os.getcwd()}")
    exit()

# --- 3. TEXT PREPROCESSING ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # HTML tag removal
    text = re.sub(r'<br\s*/?>', ' ', text)
    # Punctuation and numbers removal
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    # Lemmatize and stop-word removal
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)

print("Cleaning 50,000 reviews... (This takes 30-60 seconds)")
df['review'] = df['review'].apply(clean_text)

# Convert labels to 1 and 0
df['sentiment'] = df['sentiment'].str.strip().str.lower().map({'positive': 1, 'negative': 0})
print("✅ Text preprocessing completed")

# --- 4. MODELING ---
print("Vectorizing data...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- 5. EVALUATION ---
y_pred = model.predict(X_test)
print("\n" + "="*40)
print(f"MODEL ACCURACY: {accuracy_score(y_test, y_pred):.2%}")
print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, y_pred))
print("="*40)

# --- 6. SAVE MODEL FOR DJANGO ---
print("\n💾 Saving model for Django integration...")
try:
    joblib.dump(model, 'sentiment_project/analyzer/sentiment_model.joblib')
    joblib.dump(tfidf, 'sentiment_project/analyzer/tfidf_vectorizer.joblib')
    print("✅ Model saved as 'sentiment_model.joblib'")
    print("✅ Vectorizer saved as 'tfidf_vectorizer.joblib'")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# --- 7. TEST A REVIEW ---
def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    res = model.predict(vec)[0]
    return "POSITIVE 😊" if res == 1 else "NEGATIVE 😞"

print("\n" + "="*40)
Test = input("Enter a movie review to test sentiment: ")
if Test.strip():
    print(f"\n📊 Result: {predict_sentiment(Test)}")
print("="*40)