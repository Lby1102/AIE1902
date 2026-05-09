# CAPTCHA Recognition with Qwen2.5-VL Fine-tuning

A complete pipeline for collecting, labeling, and fine-tuning a vision-language model (Qwen2.5-VL) to recognize 4-character CAPTCHA images.

## Project Overview

This project demonstrates a full workflow for building a CAPTCHA recognition system using fine-tuned Vision-Language Models (VLM). The pipeline consists of three main stages:

1. **Data Collection** - Automated collection of CAPTCHA images from target server
2. **Data Labeling** - Multi-stage labeling using OCR and LLM APIs
3. **Model Fine-tuning** - Fine-tuning Qwen2.5-VL model with LLaMA-Factory

## Project Structure

```
AIE1902final/
├── data collection&labelling/
│   └── Data_Collection&Labelling/
│       ├── data_collection.ipynb    # CAPTCHA image collection script
│       ├── captcha_ocr.ipynb        # Preliminary labeling with ddddocr
│       ├── label.ipynb              # Precise labeling with Kimi API
│       ├── .env                     # API keys configuration (not tracked)
│       └── .env.example             # Example environment file
│
├── fine-tuning/
│   ├── translation&fine-tuning.py   # Data format conversion & fine-tuning guide
│   ├── evaluation.py                # Model evaluation script
│   ├── web_demo.py                  # Gradio web demo interface
│   ├── qwen2.5_data.json            # Training dataset in Alpaca format
│   └── eval_results.txt             # Evaluation results
│
└── requirements.txt                 # Python dependencies
```

## Module Details

### 1. Data Collection Module (`data collection&labelling/`)

#### `data_collection.ipynb`
- **Purpose**: Collects CAPTCHA images from a target server
- **Functionality**:
  - Downloads SVG CAPTCHA images from specified URL
  - Converts SVG to PNG format using `svglib` and `reportlab`
  - Uses multi-threading (10 workers) for efficient batch downloading
  - Automatically skips already downloaded images

#### `captcha_ocr.ipynb`
- **Purpose**: Generates preliminary labels using traditional OCR
- **Functionality**:
  - Uses `ddddocr` library for initial CAPTCHA recognition
  - Multi-threaded processing for speed
  - Creates initial label files for later refinement
  - Serves as a baseline for comparison

#### `label.ipynb`
- **Purpose**: Generates high-quality labels using LLM API
- **Functionality**:
  - Uses Kimi (Moonshot AI) vision API for accurate recognition
  - Implements 40 parallel API clients for high throughput
  - Base64 encodes images for API transmission
  - Produces ground truth labels for training

#### `.env` / `.env.example`
- **Purpose**: Store API keys for Moonshot AI
- **Format**: `API_KEY_1=your_key_here` up to `API_KEY_40`

---

### 2. Fine-tuning Module (`fine-tuning/`)

#### `translation&fine-tuning.py`
- **Purpose**: Prepares training data and documents fine-tuning process
- **Functionality**:
  - Converts image-label pairs to Alpaca format JSON
  - Filters invalid labels (non-4-character, non-alphanumeric)
  - Documents LLaMA-Factory training and merge commands
- **Output**: `qwen2.5_data.json`

#### `qwen2.5_data.json`
- **Purpose**: Training dataset in Alpaca format
- **Format**: Each line contains:
  ```json
  {
    "instruction": "<image>Recognize the four characters...",
    "output": "ABCD",
    "image_path": ["/path/to/image.png"]
  }
  ```

#### `evaluation.py`
- **Purpose**: Evaluates fine-tuned model performance
- **Functionality**:
  - Loads fine-tuned Qwen2.5-VL model
  - Tests on 1000 validation images
  - Uses greedy decoding for reproducibility
  - Calculates accuracy and saves detailed results
- **Output**: `eval_results.txt`

#### `web_demo.py`
- **Purpose**: Interactive web demo using Gradio
- **Functionality**:
  - Loads fine-tuned model for inference
  - Provides web interface for CAPTCHA upload
  - Returns recognized characters instantly
  - Supports public sharing via Gradio

#### `eval_results.txt`
- **Purpose**: Stores evaluation results
- **Format**: `picture: img.png | answer: XXXX | prediction: XXXX | situation: yes/no`

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Vision-Language Model | Qwen2.5-VL-3B-Instruct |
| Fine-tuning Framework | LLaMA-Factory |
| Traditional OCR | ddddocr |
| LLM API | Kimi (Moonshot AI) |
| Web Interface | Gradio |
| Image Processing | svglib, reportlab |
| Deep Learning | PyTorch, Transformers |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/captcha-recognition.git
cd captcha-recognition

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Data Collection
```bash
# Open and run data_collection.ipynb
# This will download CAPTCHA images to ./data_collection/
```

### Step 2: Data Labeling
```bash
# Configure API keys
cp data\ collection&labelling/Data_Collection&Labelling/.env.example .env
# Edit .env with your Kimi API keys

# Run labeling (optional: run captcha_ocr.ipynb first for baseline)
# Open and run label.ipynb
```

### Step 3: Prepare Training Data
```bash
cd fine-tuning
python translation&fine-tuning.py
```

### Step 4: Fine-tuning with LLaMA-Factory
```bash
# Training
llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset my_ocr_data \
  --template qwen2_vl \
  --finetuning_type lora \
  --output_dir qwen2_5_vl_ocr_lora_3 \
  --overwrite_output_dir \
  --cutoff_len 1024 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 5.0 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_steps 100 \
  --fp16 \
  --lora_target all

# Merge LoRA weights
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter_name_or_path qwen2_5_vl_ocr_lora_3/checkpoint-xxx \
  --template qwen2_vl \
  --finetuning_type lora \
  --export_dir qwen2_5_vl_merged_model_3 \
  --export_size 5 \
  --export_device auto \
  --export_legacy_format false
```

### Step 5: Evaluation
```bash
# Update MODEL_PATH in evaluation.py
python evaluation.py
```

### Step 6: Web Demo
```bash
# Update model_path in web_demo.py
python web_demo.py
# Access at http://localhost:7861
```

## Results

The fine-tuned model achieves high accuracy on 4-character alphanumeric CAPTCHA recognition. See `eval_results.txt` for detailed per-image results.

## License

This project is for educational purposes only. Please ensure compliance with target website's terms of service when collecting CAPTCHA images.

## Acknowledgments

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - Vision-Language Model
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - Fine-tuning Framework
- [ddddocr](https://github.com/sml2h3/ddddocr) - Traditional OCR Library
- [Moonshot AI](https://moonshot.cn/) - Kimi Vision API
