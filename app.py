import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st
import pickle
import string

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the pre-trained model and vectorizer
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Page title
st.set_page_config(page_title="SMS Spam Detection", page_icon="📩", layout="centered")
st.title("📩 SMS Spam Detection Model")
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
    font-family: Arial, sans-serif;
}

h1 {
    color: #2b6777;
    font-weight: bold;
    margin-bottom: 1rem;
}

div.stButton button {
    background-color: #2b6777;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    padding: 10px 20px;
    transition: all 0.3s ease;
}

div.stButton button:hover {
    background-color: #52ab98;
    color: #fff;
    cursor: pointer;
}

div.stTextInput input {
    font-size: 16px;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #ccc;
    width: 100%;
}

h2 {
    margin-top: 2rem;
    color: #2b6777;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

input_sms = st.text_input("Enter an SMS below to check if it's Spam or Not Spam.", placeholder="Type your message here...")

# Predict button
if st.button('Predict'):

    # Preprocess the input
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tk.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Display result
    if result == 1:
        st.header("🚨 Spam")
    else:
        st.header("✅ Not Spam")

