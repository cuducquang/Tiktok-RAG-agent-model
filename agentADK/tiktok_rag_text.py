from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import time

# Use local LM Studio API with OpenAI-like interface
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Vector index and associated texts
text_chunks = []
vectors = []

# Scrape TikTok profile for text data (improved version)
def scrape_profile_text(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        time.sleep(5) # Wait for page to load

        # Get profile name
        try:
            name = page.locator("h2").first.inner_text()
        except Exception as e:
            print("Name not found:", e)
            name = ""

        # Get username
        try:
            username = page.locator("h1").first.inner_text()
        except Exception as e:
            print("Username not found:", e)
            username = ""

        # Get bio
        try:
            bio = page.locator("div:below(h1)").nth(0).inner_text()
        except Exception as e:
            print("Bio not found:", e)
            bio = ""

        # Get captions from video descriptions
        captions = []
        try:
            time.sleep(3)
            page.mouse.wheel(0, 2500)
            time.sleep(2)
            video_descs = page.locator("div[data-e2e='browse-video-desc']")
            count = video_descs.count()
            for i in range(min(count, 5)):
                cap = video_descs.nth(i).inner_text()
                captions.append(cap)
        except Exception as e:
            print("Captions not found:", e)

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

# Ask local LLM (DeepSeek via LM Studio API)
def ask_llm(question, retrieved_docs, image_analysis=None):
    context = "\n\n".join(retrieved_docs)
    image_note = f"\n\nImage analysis also shows: {image_analysis}" if image_analysis else ""

    prompt = f"""
You are an expert in analyzing TikTok accounts.
Below is information extracted from a TikTok profile (including bio, username, captions, and possibly image analysis).

{context}{image_note}

Question: {question}
Answer clearly Yes or No and explain briefly.
"""

    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528-qwen3-8b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content