import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="News Classification Pro",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #4a4a4a;
        background-color: #1f2228;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: 1px solid #ffffff;
    }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1f2228 0%, #161a1e 100%);
        border-left: 5px solid #ff4b4b;
        margin-top: 20px;
    }
    .confidence-meter {
        height: 10px;
        background-color: #333;
        border-radius: 5px;
        overflow: hidden;
        margin-top: 10px;
    }
    .confidence-level {
        height: 100%;
        background-color: #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Model Loading ---
MODEL_NAME = "asimtaseer/news-classifier"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/news.png", width=150)
    st.title("Project Info")
    st.markdown("""
    ### 🚀 News Classifier
    An advanced NLP system built to categorize news articles into four distinct topics using deep learning.
    
    **Model Details:**
    - **Architecture:** Transformer
    - **Dataset:** AG News
    - **Host:** Hugging Face
    
    **Supported Categories:**
    - 🌍 World
    - ⚽ Sports
    - 💼 Business
    - 💻 Science/Technology
    
    ---
    **Author:** [Asim Taseer](https://asimtaseer.unaux.com)
    """)
    st.info("Paste any news headline or article to see the AI in action!")

# --- Main App ---
st.title("📰 News Topic Classification System")
st.markdown("#### Classify global news into categories instantly using Artificial Intelligence.")
st.write("This application uses a transformer-based model trained on the AG News dataset to provide accurate topic classification with confidence scores.")

# Input area
input_text = st.text_area("Paste news text here:", placeholder="Enter a news article, headline, or paragraph...", height=200)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    classify_btn = st.button("🚀 Classify News")

if classify_btn:
    if not input_text.strip():
        st.warning("⚠️ Please enter some text to classify.")
    else:
        with st.spinner("🧠 AI is analyzing the content..."):
            try:
                # Load model components
                tokenizer, model = load_model()
                
                # Perform inference
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = F.softmax(outputs.logits, dim=1)
                    confidence, prediction = torch.max(probabilities, dim=1)
                
                # Mapping labels (AG News categories)
                # 0 -> World, 1 -> Sports, 2 -> Business, 3 -> Sci/Tech
                categories = {
                    0: "World 🌍",
                    1: "Sports ⚽",
                    2: "Business 💼",
                    3: "Science/Technology 💻"
                }
                
                predicted_label = categories.get(prediction.item(), "Unknown")
                score = confidence.item() * 100
                
                # Artificial sleep for smoother UI experience
                time.sleep(0.5)
                
                # Display Results
                st.markdown(f"""
                <div class="result-card">
                    <h3 style="margin:0;">Prediction Result:</h3>
                    <h1 style="color:#ff4b4b; margin: 10px 0;">{predicted_label}</h1>
                    <p style="margin-bottom: 5px;"><b>Confidence Score:</b> {score:.2f}%</p>
                    <div class="confidence-meter">
                        <div class="confidence-level" style="width: {score}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Success balloons
                if score > 80:
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ An error occurred during classification: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>Developed with ❤️ by Asim Taseer | Powered by Hugging Face & Streamlit</p>",
    unsafe_allow_html=True
)
