
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "princy-21/sentilens-distilbert"

st.set_page_config(
    page_title="SentiLens | Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

st.title("SentiLens")
st.subheader("AI-Powered Movie Review Sentiment Analysis")
st.write(
    "Analyze a movie review using a DistilBERT model fine-tuned on the IMDB dataset."
)

try:
    with st.spinner("Loading the model..."):
        tokenizer, model = load_model()
except Exception as error:
    st.error("The model could not be loaded. Please try again later.")
    st.exception(error)
    st.stop()

review = st.text_area(
    "Enter a movie review",
    placeholder="Example: The acting was excellent and the story was engaging...",
    height=180,
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing review..."):
            inputs = tokenizer(
                review,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )

            with torch.no_grad():
                outputs = model(**inputs)

            probabilities = torch.softmax(outputs.logits, dim=1)[0]
            predicted_id = int(torch.argmax(probabilities).item())
            sentiment = model.config.id2label[predicted_id]

            negative = float(probabilities[0].item())
            positive = float(probabilities[1].item())

        if sentiment.upper() == "POSITIVE":
            st.success(f"Prediction: {sentiment}")
        else:
            st.error(f"Prediction: {sentiment}")

        st.write("### Class probabilities")
        st.metric("Positive", f"{positive:.2%}")
        st.progress(positive)
        st.metric("Negative", f"{negative:.2%}")
        st.progress(negative)

st.divider()
st.caption(
    "Model: DistilBERT fine-tuned on IMDB. "
    "Predictions reflect the model's output and may be incorrect."
)
