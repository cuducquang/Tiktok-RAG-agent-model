# TikTok Travel Profile Analyzer (RAG + CLIP + DeepSeek)

This project is a browser-based intelligent agent that analyzes TikTok accounts to determine whether a profile is primarily focused on **travel content**. It combines image and text processing through the following components:

* **Playwright**: Automates browser interaction with TikTok profiles
* **CLIP (OpenAI)**: Analyzes profile screenshots and thumbnails to detect visual cues related to travel
* **RAG (Retrieval-Augmented Generation)**: Uses semantic search on bio, username, and video captions
* **LLM (DeepSeek via LM Studio)**: Provides a final decision (Yes/No + explanation) using all combined information

---

## How to Set Up & Run
### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the LLM model

* Use **[LM Studio]** to download the model:

  ```
  ```

deepseek/deepseek-r1-0528-qwen3-8b

````
- Start the LM Studio API server (ensure it's running at `http://127.0.0.1:1234`)
- No authentication required; the app will call it via OpenAI-compatible API

### 3. Run the agent
```bash
python run.py
````

This script will:

* Visit each TikTok profile
* Screenshot thumbnails and classify them using CLIP
* Extract bio, username, and captions using Playwright
* Perform semantic analysis using FAISS + SentenceTransformer
* Ask DeepSeek to give a final decision based on all data

### 5. Results

* Results will be printed in the terminal
* Results will also be saved to `output.txt`

---
