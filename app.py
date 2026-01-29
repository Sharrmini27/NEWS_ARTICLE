import streamlit as st
from transformers import pipeline
from newspaper import Article
import time
import nltk

# Fix for NLTK errors in cloud environments
@st.cache_resource
def download_nltk():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except Exception as e:
        st.error(f"NLTK Download Error: {e}")

download_nltk()

# 1. Page Configuration
st.set_page_config(page_title="AI News Summarizer", page_icon="📝")

# 2. Optimized Model Loading (Bypasses KeyError & Memory Limits)
@st.cache_resource
def load_summarizer():
    # We use DistilBART: It is much faster and fits in Streamlit's 1GB RAM limit.
    # We explicitly define 'task' and 'model' to prevent KeyError.
    return pipeline(task="summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

st.title("🤖 AI-Powered News Summarizer")
st.markdown("Enter a news URL below to generate a concise AI summary.")

# 3. User Input
url = st.text_input("🔗 Paste News Article URL here:", placeholder="https://www.channelnewsasia.com/...")

if st.button("Generate Summary"):
    if url:
        try:
            with st.spinner('AI is reading and summarizing...'):
                start_time = time.time()
                
                # Fetch and Parse
                article = Article(url)
                article.download()
                article.parse()
                
                if not article.text:
                    st.error("❌ Could not extract text. Try a different news site.")
                else:
                    # Execute Summarization
                    # We limit input to 3000 chars to stay within model limits
                    summary_output = summarizer(article.text[:3000], max_length=130, min_length=30, do_sample=False)
                    summary_text = summary_output[0]['summary_text']
                    
                    duration = round(time.time() - start_time, 2)
                    
                    # 4. Display Results
                    st.subheader(f"Title: {article.title}")
                    st.success(summary_text)
                    
                    # Performance Metrics
                    orig_len = len(article.text.split())
                    summ_len = len(summary_text.split())
                    reduction = round((1 - (summ_len / orig_len)) * 100, 1)
                    
                    st.write("---")
                    st.info(f"**Performance:** {orig_len} words ➡️ {summ_len} words ({reduction}% reduction) | ⏱ {duration}s")
                    
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter a URL first.")

# Sidebar
st.sidebar.title("Project Details")
st.sidebar.write("**Model:** DistilBART (Optimized)")
st.sidebar.write("**Student Task:** JIE43303 NLP")
