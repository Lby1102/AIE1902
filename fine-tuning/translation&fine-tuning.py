import os
import json

image_dir = "./dataset/data_collection"
label_dir = "./dataset/data_label"
output_file = "qwen2.5_data.json"

allowed_letter = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
dataset = []

for file_name in os.listdir(image_dir):
    if file_name.endswith(".png"):
        base_name = os.path.splitext(file_name)[0] # get file name
        label_path = os.path.join(label_dir,f"{base_name}.txt") # corresponding label name
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                label_text = f.read().strip()
        
        # data cleaning
        if len(label_text)!=4:
            continue # skip non-4-digit
        flag = True
        for char in label_text:
            if char not in allowed_letter:
                flag = False
                break
        if flag == False:
            continue # skip those containing non-digits or letters
        data_line = { # use Alpaca format
            "instruction": "<image>Recognize the four characters in the image, directly output the four characters, do not output any other text",
            "output": label_text,
            "image_path": [os.path.abspath(os.path.join(image_dir, file_name))]
            }
        dataset.append(data_line)

with open(output_file,'w',encoding='utf-8') as f:
    for item in dataset:
        line = json.dumps(item,ensure_ascii=False)
        f.write(line+"\n")

print("okk!")


#for fine-tuning we use LLaMA-Factory
'''
fine-tuning command:
llamafactory-cli train --stage sft --do_train --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct --dataset my_ocr_data --template qwen2_vl --finetuning_type lora --output_dir qwen2_5_vl_ocr_lora_3 --overwrite_output_dir --cutoff_len 1024 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 1e-4 --num_train_epochs 5.0 --lr_scheduler_type cosine --logging_steps 10 --save_steps 100 --fp16 --lora_target all

model merge command:
llamafactory-cli export --model_name_or_path qwen/Qwen2.5-VL-3B-Instruct --adapter_name_or_path qwen2_5_vl_ocr_lora_3/checkpoint-x(to be modified) --template qwen2_vl --finetuning_type lora --export_dir qwen2_5_vl_merged_model_3 --export_size 5 --export_device auto --export_legacy_format false
'''
