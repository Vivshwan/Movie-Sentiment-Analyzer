# 🎬 Movie Sentiment Analyzer

A Django-based web application that analyzes movie reviews and predicts whether they are *POSITIVE* or *NEGATIVE* using Machine Learning.

## 📊 Model Performance
- *Accuracy:* ~89%
- *Algorithm:* Logistic Regression
- *Features:* TF-IDF (5000 features)
- *Dataset:* 50,000 IMDB movie reviews

## ✨ Features
- Clean and intuitive web interface
- Real-time sentiment prediction
- Professional UI with emoji feedback
- REST API endpoint for developers
- Comprehensive error handling

## 🛠️ Tech Stack
- *Backend:* Django 4.x
- *Machine Learning:* Scikit-learn, NLTK
- *Frontend:* HTML5, CSS3
- *Data Processing:* Pandas, NumPy

## ⚠️ Before Running
1. Download dataset from: [Kaggle Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Save as IMDB Dataset.csv in project folder

## 📋 Prerequisites
```bash
Python 3.8+
pip package manager

🔧 Installation & Setup
1. Clone the repository
bash
git clone https://github.com/Vivshwan/sentiment-analyzer.git
cd sentiment-analyzer

2. Create virtual environment (recommended)
bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate

3. Install dependencies
bash
pip install -r requirements.txt

4. Download the trained model files
Option A: Download from Google Drive (Recommended)

Download sentiment_model.joblib and tfidf_vectorizer.joblib from: [ADD YOUR GOOGLE DRIVE LINK HERE]

Place both files in the analyzer/ folder

Option B: Train the model yourself

bash
python main.py
Then copy the generated .joblib files to the analyzer/ folder

5. Run the Django application
bash
cd sentiment_project
python manage.py migrate
python manage.py runserver
6. Open your browser
Navigate to: http://127.0.0.1:8000
