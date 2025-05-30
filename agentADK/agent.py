from browser_tool import capture_tiktok_thumbnails
from analysis_tool import analyze_images_with_clip
from tiktok_rag_text import scrape_profile_text, build_document_text, add_to_index, build_faiss_index, query_semantic, ask_llm

def process_profile(profile_url: str):
    print(f"[+] Processing: {profile_url}")
    
    # Image analysis with CLIP
    screenshots = capture_tiktok_thumbnails(profile_url)
    img_result = analyze_images_with_clip(screenshots) if screenshots else "No screenshots."

    # Text RAG with FAISS + LLM
    # profile = scrape_profile_text(profile_url)
    # doc_text = build_document_text(profile)
    # add_to_index(doc_text)
    # index = build_faiss_index()
    # retrieved_docs = query_semantic("Is this account about travel?", index)
    # llm_result = ask_llm("Is this account about travel?", retrieved_docs)

    return {
        "profile_url": profile_url,
        "image_analysis": img_result,
        # "llm_analysis": llm_result
    }
