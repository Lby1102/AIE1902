# CAPTCHA Recognition

A CAPTCHA recognition system using both CNN and fine-tuned Qwen2.5-VL models.

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

**Data Collection Module**: Downloads CAPTCHA images from server and labels them using OCR and LLM APIs.

**Fine-tuning Module**: Fine-tunes Qwen2.5-VL-3B using LLaMA-Factory with LoRA, includes evaluation and demo.

**CNN Module**: Traditional CNN approach with custom architecture for 4-character recognition.

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
