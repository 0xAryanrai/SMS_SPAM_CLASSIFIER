import streamlit as st
import pickle

# Load the saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
countvectorizer = pickle.load(open("countvectorizer.pkl", "rb"))

# -------------------------------
# If you used a preprocessing function during training,
# paste it here and uncomment the line below.
# Otherwise, leave it as it is.
# -------------------------------

def transform_text(text):
    # Replace this with your preprocessing function
    return text

# -------------------------------

# Streamlit page settings
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message below to check whether it is **Spam** or **Not Spam**.")

# User input
message = st.text_area("Enter your message")

# Prediction button
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess the text
        transformed_message = transform_text(message)

        # Convert text into numerical features
        vector_input = countvectorizer.transform([transformed_message])

        # Predict
        prediction = model.predict(vector_input)

        # Display result
        if prediction[0] == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")