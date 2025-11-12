# streamlit_pegasus_app.py

import streamlit as st
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import html
import re

# ----- Configuration -----
MODEL_DIR = r"D:\DBDA\News Summarization\Models\pegasus_english_finetuned\final_model"
 
MAX_INPUT_LENGTH = 1024
MAX_SUMMARY_LENGTH = 100
BEAM_SIZE = 4

# ----- Helper Functions -----
def clean_text(text):
    """
    Clean input news article text by:
    - Decoding HTML entities
    - Removing control characters, zero-width spaces
    - Removing emojis and some special characters
    - Normalizing whitespace
    """
    if not isinstance(text, str):
        return ""
    try:
        text = html.unescape(text)
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = re.sub(r'[\u200B-\u200D\uFEFF\x00-\x1F\x7F-\x9F]', '', text)
        text = re.sub(r'[#$@%&*{}<>\\^~`|]', '', text)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        st.warning(f"Warning: error during text cleaning: {e}")
        return ""

@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_DIR)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_DIR)
    model = model.to(device).eval()
    return tokenizer, model, device

def generate_summary(article_text, tokenizer, model, device):
    cleaned_article = clean_text(article_text)
    
    if len(cleaned_article.split()) < 10:
        return None, "Input text is too short to generate a meaningful summary."
    
    try:
        inputs = tokenizer(
            cleaned_article,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            padding="longest"
        )
        if inputs.input_ids.min() < 0 or inputs.input_ids.max() >= tokenizer.vocab_size:
            return None, ("Input contains tokens outside the model’s vocabulary. " 
                          "Please modify your input and try again.")
        
        inputs = inputs.to(device)
        with torch.no_grad():
            summary_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=MAX_SUMMARY_LENGTH,
                num_beams=BEAM_SIZE,
                early_stopping=True
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary, None
    
    except Exception as e:
        return None, f"Error during summary generation: {str(e)}"

# ----- Streamlit Interface -----
st.set_page_config(page_title="Pegasus News Summarizer", page_icon="📰", layout="centered")

st.title("📰 News Summarizer")
st.write("Paste a news article text below and click *Summarize* to generate its summary.")

input_text = st.text_area("News Article", height=300, placeholder="Paste or type your article here...")

if st.button("Summarize"):
    if not input_text.strip():
        st.warning("Please enter the news article text you want to summarize.")
    else:
        tokenizer, model, device = load_model_and_tokenizer()
        with st.spinner("Generating summary, please wait..."):
            summary, warning_msg = generate_summary(input_text, tokenizer, model, device)
        if warning_msg:
            st.warning(warning_msg)
        elif summary:
            st.subheader("Summary")
            st.write(summary)
        else:
            st.error("Unknown error occurred.")
