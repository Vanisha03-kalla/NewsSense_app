# Streamlit App 
import streamlit as st
import joblib

# Load trained model + vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Optional: map numeric labels to strings for UI
label_map = {0: 'FAKE', 1: 'REAL'}

# Streamlit UI setup
st.set_page_config(page_title="Fake News Classifier 📰", page_icon="🧠", layout="centered")
st.title("📰 Fake News Classifier")
st.write("Check if a news article sounds **Real or Fake** using NLP + Machine Learning!")

# User input
user_input = st.text_area("📝 Paste a news article or headline below:", height=200)

if st.button("Analyze"):
    if len(user_input.strip()) == 0:
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        # Transform input and make prediction
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)[0]        # 0 or 1
        prediction_proba = model.predict_proba(transformed_text)[0]

        # Dynamic confidence
        classes = model.classes_   # e.g., [0, 1]
        pred_index = list(classes).index(prediction)
        confidence = prediction_proba[pred_index] * 100

        # Map numeric label to string
        pred_label = label_map[prediction]

        # Show result
        if pred_label == 'REAL':
            st.success(f"✅ This news seems **REAL**.\nConfidence: {confidence:.2f}%")
        else:
            st.error(f"🚨 This news seems **FAKE**.\nConfidence: {confidence:.2f}%")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "Built with ❤️ by <b>Vanisha Kalla</b><br>"
    "💻 <a href='https://github.com/Vanisha03-kalla' target='_blank'>Check out my GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
