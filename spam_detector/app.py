import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords and punkt if not already present
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stemmer
ps = PorterStemmer()

# Load vectorizer and model
count_vectoriser = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Streamlit UI
st.title("📧 Email / SMS Spam Classifier")

inp_sms = st.text_input("Enter the message:")

# Define transformation function
def transform_data(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Run only if user provides input
if inp_sms:
    # 1. Preprocess
    transformed_text = transform_data(inp_sms)

    # 2. Vectorize
    vector = count_vectoriser.transform([transformed_text])

    # 3. Predict
    result = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][result] if hasattr(model, "predict_proba") else None

    # 4. Display
    if result == 1:
        st.header("🚫 Spam")
    else:
        st.header("✅ Not Spam")

    # Optional: Show probability
    if probability is not None:
        st.write(f"Prediction confidence: {probability:.2f}")

    # Optional: Show transformed text (helpful for debugging)
    st.write(f"Transformed text: `{transformed_text}`")
else:
    st.info("Please enter a message above to classify it.")
