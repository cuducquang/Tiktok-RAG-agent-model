from agent import process_profile

test_profiles = [
    "https://www.tiktok.com/@tainamdu94",
    "https://www.tiktok.com/@manhmytrainghiem",
    "https://www.tiktok.com/@quality.trip",
    "https://www.tiktok.com/@mi_iuoi00"
]

if __name__ == "__main__":
    with open("output.txt", "w", encoding="utf-8") as f:
        for url in test_profiles:
            line = "=" * 50
            print(line)
            f.write(line + "\n")

            log = f"[+] Processing: {url}"
            print(log)
            f.write(log + "\n")

            result = process_profile(url)

            img_result = result['image_analysis']
            llm_result = result['llm_analysis']

            print("\n[Image Analysis Result]:")
            print(img_result)
            f.write("\n[Image Analysis Result]:\n" + img_result + "\n")

            print("\n[Text RAG Result]:")
            print(llm_result)
            f.write("\n[Text RAG Result]:\n" + llm_result + "\n")

            print("\n")
            f.write("\n")