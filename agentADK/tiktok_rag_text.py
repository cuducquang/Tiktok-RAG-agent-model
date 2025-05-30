from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize embedding model and LLM client
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Vector index and associated texts
text_chunks = []
vectors = []

# Scrape TikTok profile for text data
def scrape_profile_text(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        page.wait_for_selector("h2", timeout=10000)

        try:
            bio = page.locator('div[data-e2e="user-bio"]').inner_text()
        except:
            bio = ""

        try:
            name = page.locator('h2[data-e2e="user-title"]').inner_text()
        except:
            name = ""

        try:
            username = page.locator('h1[data-e2e="user-username"]').inner_text()
        except:
            username = ""

        captions = []
        try:
            page.wait_for_selector('div[data-e2e="browse-video-desc"]', timeout=5000)
            caption_elements = page.locator('div[data-e2e="browse-video-desc"]').all()
            for el in caption_elements[:5]:
                captions.append(el.inner_text())
        except:
            pass

        browser.close()

        return {
            "url": url,
            "name": name,
            "username": username,
            "bio": bio,
            "captions": captions
        }

# Build one document string from profile
def build_document_text(profile_data):
    parts = [
        f"Name: {profile_data['name']}",
        f"Username: {profile_data['username']}",
        f"Bio: {profile_data['bio']}",
        "Caption videos:",
    ] + profile_data["captions"]
    return "\n".join(parts)

# Add document to index
def add_to_index(doc_text):
    vector = embedder.encode(doc_text)
    text_chunks.append(doc_text)
    vectors.append(vector)

# Create FAISS index
def build_faiss_index():
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))
    return index

# Retrieve top-k related chunks
def query_semantic(question, index, top_k=3):
    q_vec = embedder.encode(question).reshape(1, -1)
    D, I = index.search(q_vec, top_k)
    return [text_chunks[i] for i in I[0]]

# Ask LLM with retrieved context
def ask_llm(question, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    prompt = f"""
You are an expert of analysis tiktok's accounts. Below is ìnformation about a TikTok profile:

{context}

Question: {question}
Answer clearly Yes or No, and short brief explaination.
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
