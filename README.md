# CAPTCHA Recognition: A Comparative Study on Noisy Image Recognition

## Project Objective

This project aims to explore methods for improving model performance on noisy image recognition tasks. Using CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) as a case study, we conduct a comparative analysis between **CNN-based approaches** and **Vision Language Model fine-tuning** to investigate optimization strategies for handling noisy images.

### Key Research Questions
- How can models be optimized to better recognize characters in noisy, distorted images?
- What are the comparative advantages of traditional CNN architectures versus fine-tuned Vision Language Models for this task?
- Which approach achieves better generalization on challenging CAPTCHA images with noise, distortion, and interference?

## Project Structure

```
AIE1902final/
├── data collection&labelling/
│   └── Data_Collection&Labelling/
│       ├── data_collection.ipynb    # Download CAPTCHA images (SVG→PNG)
│       ├── captcha_ocr.ipynb        # Preliminary labeling with ddddocr
│       ├── label.ipynb              # Accurate labeling with Kimi API
│       └── .env.example             # API key configuration template
│
├── data_preprocessing/
│   └── Binarization.ipynb           # Image binarization: convert to 0/1 matrix, rename by label
│
├── fine-tuning/
│   ├── translation&fine-tuning.py   # Convert data to Alpaca format
│   ├── evaluation.py                 # Evaluate fine-tuned model accuracy
│   ├── web_demo.py                   # Gradio web demo
│   ├── qwen2.5_data.json             # Training dataset
│   └── eval_results.txt             # Evaluation results
│
├── CNN/
│   ├── train_CNN.ipynb              # Train CNN model
│   ├── predict_captcha_CNN.py       # Single image prediction
│   ├── evaluate_CNN.py             # Batch accuracy evaluation
│   └── CNN_model.pth                # Trained model weights
│
└── requirements.txt                 # All Python dependencies
```

## Module Description

### Data Collection Module
Downloads CAPTCHA images from server and labels them using OCR and LLM APIs.

### Data Preprocessing Module
Processes raw images for model training:
- **Binarization**: Converts grayscale images to binary 0/1 matrices using Otsu's thresholding
- **Renaming**: Renames files based on their label content for easy identification
- **Deduplication**: Removes duplicate files and standardizes naming to uppercase

### Fine-tuning Module
Fine-tunes Qwen2.5-VL-3B using LLaMA-Factory with LoRA, includes evaluation and demo.

### CNN Module
Traditional CNN approach with custom architecture for 4-character recognition.

## Methodology

### Approach 1: CNN-based Recognition
A custom Convolutional Neural Network architecture designed for 4-character CAPTCHA recognition, featuring:
- Multi-head output for simultaneous character prediction
- Batch normalization and dropout for regularization
- Binary thresholding preprocessing for noise reduction

### Approach 2: Vision Language Model Fine-tuning
Fine-tuning Qwen2.5-VL-3B using LLaMA-Factory with LoRA (Low-Rank Adaptation):
- Leverages pre-trained vision-language understanding
- Efficient fine-tuning with limited computational resources
- Natural language prompting for flexible inference

## Quick Start

```bash
pip install -r requirements.txt
```

For CNN model:
```bash
cd CNN
python predict_captcha_CNN.py
```

For fine-tuned VLM:
```bash
cd fine-tuning
python web_demo.py
```

## License

For educational purposes only.
