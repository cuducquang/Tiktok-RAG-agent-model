import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

categories = ["tourism", "food", "fashion", "entertainment", "office", "sport", "family", "animals"]

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=categories, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
    label_prob_pairs = list(zip(categories, probs))
    label_prob_pairs.sort(key=lambda x: x[1], reverse=True)
    return label_prob_pairs[0]

def analyze_images_with_clip(image_paths):
    results = []
    count_tourism = 0

    for image_path in image_paths:
        label, prob = classify_image(image_path)
        print(f"[+] {os.path.basename(image_path)} → {label} ({prob:.2%})")
        results.append((image_path, label, prob))

        if label == "tourism" and prob >= 0.6:
            count_tourism += 1

    if count_tourism >= len(image_paths) / 2:
        return "This account is primarily about tourism."
    else:
        return "This account is not primarily about tourism."
