import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_path = "./qwen2_5_vl_merged_model_3" 

# qwen/Qwen2.5-VL-3B-Instruct
# ./qwen2_5_vl_merged_model_3
print("uploading the model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_path)
print("finished!")

def predict(image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Recognize the four characters in the image, directly output the four character"},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    return output_text[0]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Uploading 4-character image"),
    outputs=gr.Textbox(label="result"),
    title="Qwen2.5-VL Fine-tuned Model",
    description="upload a PNG picture, the model will tell you the 4 characters in it"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)