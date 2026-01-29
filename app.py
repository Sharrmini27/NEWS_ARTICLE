import streamlit as st
from transformers import pipeline
from newspaper import Article
import time
import nltk

# 1. Setup Logic: Mandatory for newspaper3k to extract text
@st.cache_resource
def initialize_system():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except:
        pass

initialize_system()

# 2. Optimized Model Loading
# Using BART (Bidirectional and Auto-Regressive Transformers)
@st.cache_resource
def load_summarizer():
    # We use 'facebook/bart-large-cnn' as specified in your report
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# 3. Streamlit UI Design
st.set_page_config(page_title="AI News Summarizer", page_icon="📰", layout="wide")

st.title("🤖 AI-Powered News Summarizer")
st.markdown("---")

# Input Section
url = st.text_input("🔗 Paste News Article URL here:", placeholder="https://www.example.com/news...")

if st.button("Generate Summary"):
    if url:
        try:
            with st.spinner('AI is reading and distilling the article...'):
                start_time = time.time()
                
                # Step 1: Web Scraping
                article = Article(url)
                article.download()
                article.parse()
                
                if not article.text:
                    st.error("❌ Could not extract text. The website may be blocking automated access.")
                else:
                    # Step 2: Abstractive Summarization
                    # We truncate input to 1024 tokens to match model limits
                    summary_output = summarizer(article.text[:1024], max_length=130, min_length=30, do_sample=False)
                    summary_text = summary_output[0]['summary_text']
                    
                    duration = round(time.time() - start_time, 2)
                    
                    # Step 3: Display Results
                    st.subheader(f"📄 Title: {article.title}")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown("### 📝 AI Summary")
                        st.success(summary_text)
                    
                    with col2:
                        st.markdown("### 📊 Performance")
                        orig_len = len(article.text.split())
                        summ_len = len(summary_text.split())
                        reduction = round((1 - (summ_len / orig_len)) * 100, 1)
                        
                        st.metric("Reduction", f"{reduction}%")
                        st.write(f"**Original:** {orig_len} words")
                        st.write(f"**Summary:** {summ_len} words")
                        st.write(f"**Processing Time:** {duration}s")
                    
        except Exception as e:
            st.error(f"⚠️ System Error: {str(e)}")
    else:
        st.warning("⚠️ Please provide a URL first.")

# Sidebar Information
st.sidebar.title("Project Information")
st.sidebar.write("**Student:** Sharrmini Veeran (S22A0037)")
st.sidebar.write("**Course:** JIE43303 NLP")
st.sidebar.write("**Model:** BART (Abstractive)")
