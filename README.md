# ⚡ Fast Multilingual PDF Q&A Assistant

A **Streamlit-based AI application** that allows users to upload PDFs and **ask questions in English, Hindi, or Marathi** — and get instant answers directly from the document.

This tool uses **Semantic Search with Transformers** and optional **OCR (Optical Character Recognition)** to understand both text-based and scanned PDFs.

---

## 🌟 Features

✅ **Multilingual Support** — Works with English, Hindi, and Marathi documents  
✅ **Semantic Q&A** — Finds the most relevant answer from the PDF  
✅ **Scanned PDF OCR** — Automatically extracts text using Tesseract when needed  
✅ **Image Retrieval** — Returns diagrams, images, and figures when requested  
✅ **Chat-style Interface** — Keeps conversation history for context  
✅ **Fast Execution** — Lightweight multilingual model optimized for CPU  
✅ **Streamlit UI** — Clean and responsive interface  

---

## 🧠 How It Works

1. User uploads a PDF (text or scanned).
2. The app extracts:
   - **Text** using `pdfplumber`
   - **Images** embedded in the file
   - **OCR text** using `pytesseract` if no selectable text exists
3. The text is split into chunks, encoded using multilingual sentence embeddings.
4. For each user question:
   - A semantic similarity search finds the most relevant chunk.
   - The answer is displayed with confidence score.
   - If query includes “image”, diagrams from the PDF are shown.
5. Conversation history is preserved within the session.

---

## 🧰 Tech Stack

| Layer | Technology / Tool | Purpose |
|-------|-------------------|----------|
| **Frontend / UI** | Streamlit | Web interface |
| **Backend Logic** | Python | App & data flow |
| **Text Extraction** | pdfplumber | Extracts text from PDF pages |
| **Image Extraction** | pdfplumber + Pillow | Fetches embedded images |
| **OCR (Scanned PDFs)** | pytesseract + pdf2image | Converts images to text |
| **Embeddings** | SentenceTransformer (`distiluse-base-multilingual-cased-v2`) | Semantic understanding |
| **Language Detection** | langdetect | Detects English/Hindi/Marathi |
| **Search Similarity** | cosine similarity (PyTorch + SentenceTransformers) | Finds best matching answers |

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/multilingual-pdf-qa.git
cd multilingual-pdf-qa
