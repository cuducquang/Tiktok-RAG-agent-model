from playwright.sync_api import sync_playwright
import os
import time

def capture_tiktok_thumbnails(profile_url: str, save_dir: str = "screenshots") -> list:
    os.makedirs(save_dir, exist_ok=True)
    screenshots = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(profile_url)
        time.sleep(10)

        for _ in range(5):
            page.mouse.wheel(0, 2500)
            time.sleep(5)

        img_path = os.path.join(save_dir, f"{profile_url.split('/')[-1]}_full.png")
        page.screenshot(path=img_path, full_page=True)
        screenshots.append(img_path)

        browser.close()
    return screenshots