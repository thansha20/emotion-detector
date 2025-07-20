import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Sample dataset
data = {
    'text': [
        'I am so happy today!',
        'I feel really sad and alone.',
        'Why are you shouting at me?',
        'This is the best day ever!',
        'I miss my family so much.',
        'Stop bothering me!',
        'I\'m feeling joyful and excited.',
        'I’m exhausted and feel low.',
        'You always ignore me!',
        'Life is beautiful.',
        'I can\'t stop crying.',
        'Leave me alone!'
    ],
    'emotion': [
        'Happy', 'Sad', 'Angry', 'Happy', 'Sad', 'Angry',
        'Happy', 'Sad', 'Angry', 'Happy', 'Sad', 'Angry'
    ]
}

# Preprocessing
df = pd.DataFrame(data)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# Model training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['emotion']
model = MultinomialNB()
model.fit(X, y)

# Streamlit app
st.title("🧠 Emotion Detector from Text")

user_input = st.text_input("Enter a sentence:")

if st.button("Detect Emotion"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    st.success(f"Detected Emotion: **{prediction}**")
