# analyzer/views.py
from django.shortcuts import render
from django.http import JsonResponse
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import json

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer
MODEL_PATH = os.path.join(BASE_DIR, 'sentiment_model.joblib')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.joblib')

# Load once when server starts
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)

# Initialize text processing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<br\s*/?>', ' ', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase and split
    text = text.lower().split()
    # Lemmatize and remove stopwords
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)


def predict_sentiment(review_text):
    """Predict sentiment for a given review"""
    try:
        # Clean and vectorize
        cleaned_review = clean_text(review_text)
        review_vector = tfidf.transform([cleaned_review])

        # Predict
        prediction = model.predict(review_vector)[0]

        return {
            'sentiment': 'POSITIVE' if prediction == 1 else 'NEGATIVE',
            'emoji': '😊' if prediction == 1 else '😞',
            'confidence': '~89%'
        }
    except Exception as e:
        return {
            'sentiment': 'ERROR',
            'emoji': '⚠️',
            'error': str(e)
        }


def index(request):
    """Home page - handles form submission"""
    result = None

    if request.method == 'POST':
        review_text = request.POST.get('review', '')
        if review_text.strip():
            result = predict_sentiment(review_text)
            result['review'] = review_text

    return render(request, 'analyzer/index.html', {'result': result})


def api_analyze(request):
    """API endpoint for AJAX calls"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            review_text = data.get('review', '')
            result = predict_sentiment(review_text)
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Only POST requests allowed'}, status=405)