import streamlit as st
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from io import BytesIO
import re
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from langdetect import detect

# Optional: path to Tesseract (update if not auto-detected)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Multilingual PDF Assistant", layout="wide")
st.title("🌐 Multilingual PDF Q&A Assistant (English | हिंदी | मराठी)")

# Load models (cached)
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("l3cube-pune/indic-sentence-similarity-sbert")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return embed_model, summarizer

embed_model, summarizer = load_models()

# OCR extraction (for scanned PDFs)
def extract_text_with_ocr(pdf_bytes):
    pages = convert_from_bytes(pdf_bytes)
    text = ""
    for p in pages:
        text += pytesseract.image_to_string(p, lang="eng+hin+mar") + "\n"
    return text

# Extract text & images from PDF
def extract_text_images(pdf_file):
    text = ""
    images = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
            for img in page.images:
                x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"]
                image_obj = page.within_bbox((x0, top, x1, bottom)).to_image(resolution=150)
                img_bytes = BytesIO()
                image_obj.save(img_bytes, format="PNG")
                images.append((i + 1, img_bytes.getvalue()))
    return text.strip(), images

# Chunk text for semantic search
def make_chunks(text, max_words=120):
    sents = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s.strip()) > 20]
    chunks, cur, count = [], [], 0
    for s in sents:
        wc = len(s.split())
        if count + wc > max_words and cur:
            chunks.append(" ".join(cur))
            cur, count = [s], wc
        else:
            cur.append(s)
            count += wc
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# Semantic search
def semantic_search(query, chunks, model, top_k=3):
    query_emb = model.encode(query, convert_to_tensor=True)
    chunk_emb = model.encode(chunks, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, chunk_emb)[0]
    top_vals, top_idx = torch.topk(scores, k=min(top_k, len(chunks)))
    results = [(float(v), chunks[i]) for v, i in zip(top_vals, top_idx)]
    return results

# Summarization helper
def summarize_text(text):
    if len(text.split()) < 50:
        return "Text too short to summarize."
    summary = summarizer(text[:1500], max_length=120, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Sidebar: Settings
st.sidebar.header("Settings ⚙️")
max_words = st.sidebar.slider("Chunk size (words)", 80, 300, 120)
threshold = st.sidebar.slider("Answer confidence threshold", 0.2, 0.8, 0.35)
show_summary = st.sidebar.checkbox("Auto-generate PDF summary", value=True)
show_page_num = st.sidebar.checkbox("Show page numbers for answers", value=True)

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")
if uploaded_file:
    st.success("✅ PDF uploaded successfully!")
    pdf_bytes = uploaded_file.read()
    text, images = extract_text_images(BytesIO(pdf_bytes))

    if not text.strip():
        st.warning("No text detected — using OCR (this might take longer)...")
        text = extract_text_with_ocr(pdf_bytes)

    # Display summary
    if show_summary:
        st.subheader("📝 Summary of PDF")
        summary = summarize_text(text)
        st.write(summary)

    # Create searchable chunks
    chunks = make_chunks(text, max_words=max_words)

    st.subheader("💬 Ask questions about your PDF")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask your question here (English, Hindi, or Marathi):")

    if user_query:
        lang = detect(user_query)
        st.caption(f"🈯 Detected language: {lang.upper()}")

        # Check if user asks about images
        if any(word in user_query.lower() for word in ["image", "diagram", "picture", "photo", "figure"]):
            if images:
                st.subheader("🖼️ Images found in PDF:")
                for page_num, img_bytes in images:
                    image = Image.open(BytesIO(img_bytes))
                    st.image(image, caption=f"Page {page_num}", use_container_width=True)
            else:
                st.error("No images found in this PDF.")
        else:
            results = semantic_search(user_query, chunks, embed_model, top_k=3)
            best_score, best_text = results[0]
            if best_score >= threshold:
                st.session_state.chat_history.append(("user", user_query))
                st.session_state.chat_history.append(("assistant", best_text))
                st.success(f"Answer (confidence {best_score:.2f}):")
                st.write(best_text)
                if show_page_num:
                    for idx, ch in enumerate(chunks):
                        if best_text in ch:
                            st.caption(f"📄 Found in chunk #{idx + 1}")
                            break
            else:
                st.error("❌ No relevant answer found.")

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")
        for role, text in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**🧑 You:** {text}")
            else:
                st.markdown(f"**🤖 Assistant:** {text}")
