import os
import torch
import json
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/home/qixiao/LLaMA-Factory/qwen2_5_vl_merged_model_3" 
IMAGE_DIR = "/home/qixiao/test/data_collection"
LABEL_DIR = "/home/qixiao/test/data_label2"

TEST_LIMIT = 1000 
ALLOWED_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def is_valid_label(text):
    if len(text) != 4: 
        return False
    for char in text:
        if char not in ALLOWED_CHARS: return False
    return True

def run_evaluation():
    print(f"Uploading the model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png'))]
    
    results = []
    correct_count = 0
    total_valid_count = 0
    
    print(f"Starting the test...")
    pbar = tqdm(total=TEST_LIMIT)

    for img_name in image_files:
        if total_valid_count >= TEST_LIMIT:
            break

        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")

        if not os.path.exists(label_path): continue
        with open(label_path, 'r', encoding='utf-8') as f:
            gt_label = f.read().strip()

        if not is_valid_label(gt_label): continue

        total_valid_count += 1
        messages = [{"role": "user", "content": [
            {"type": "image", "image": os.path.join(IMAGE_DIR, img_name)},
            {"type": "text", "text": "Recognize the four characters in the image, directly output the four characters, do not output any other text"}
        ]}]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            # Using greedy search (do_sample=False)
            generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            prediction = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            
            prediction_cleaned = "".join(prediction.split()).strip()

        is_correct = (prediction_cleaned.upper() == gt_label.upper())
        if is_correct:
            correct_count += 1
        
        results.append(f"picture: {img_name} | answer: {gt_label} | prediction: {prediction_cleaned} | situation: {'yes' if is_correct else 'no'}")

        if total_valid_count <= 10:
            print(f"\n[sample {total_valid_count}] answer: {gt_label} -> model: {prediction_cleaned} ({'yes' if is_correct else 'no'})")

        pbar.update(1)

    pbar.close()

    with open("eval_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"\nfinished! Accuracy rate: {(correct_count/total_valid_count)*100:.2f}%")

if __name__ == "__main__":
    run_evaluation()