import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import string
import os


captcha_len = 4
characters = string.digits + string.ascii_uppercase
num_classes = len(characters)

height = 40
width = 95
learning_rate = 0.001
epochs = 50
batch_size = 64

char_to_idx = {c: i for i, c in enumerate(characters)}
idx_to_char = {i: c for i, c in enumerate(characters)}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

class CaptchaCNN(nn.Module):
    def __init__(self) :
        super(CaptchaCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 32,
                kernel_size = 3,
                padding = 1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
        )

        self._feature_size = self._get_feature_size()

        self.fc_shared = nn.Sequential(
            nn.Linear(self._feature_size, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2)
        )

        self.fc_heads = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(captcha_len)
        ])

    def _get_feature_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, height, width)
            out = self.features(dummy)
            return out.view(1, -1).size(1)

    def forward(self, x) :

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.fc_shared(x)

        outputs = [head(x) for head in self.fc_heads]

        return outputs

def load_model():
    model = CaptchaCNN().to(DEVICE)
    script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(script_path)
    model_path = os.path.join(current_dir, 'CNN_model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Successfully found the model")
    else :
        print("Model not found")
    return model

def process_batch(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (95, 40))

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    matrix_01 = (binary > 127).astype(np.float32)

    return matrix_01

def predict(model, image_path):
    image = process_batch(image_path)
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
    prediction = ""
    for out in outputs:
        char_idx = torch.argmax(out, dim = 1).item()
        prediction += idx_to_char[char_idx]

    return prediction

model = load_model()

def calculate_accuracy(model, total_count=5000):
    script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(script_path)

    correct_predictions = 0
    correct_chars = 0
    processed_count = 0
    total_chars_count = 0

    print(f"Starting test, total data points: {total_count}...")

    for i in range(1, total_count + 1):

        img_name = f"img{i}.png"
        txt_name = f"img{i}.txt"
        img_path = os.path.join(current_dir, 'data', img_name)
        txt_path = os.path.join(current_dir, 'data', txt_name)

        if os.path.exists(img_path) and os.path.exists(txt_path):
            prediction = str(predict(model, img_path)).strip().upper()

            with open(txt_path, 'r', encoding='utf-8') as f:
                ground_truth = f.read().strip().upper()

            if len(ground_truth) != captcha_len or len(prediction) != captcha_len:
                continue

            if prediction == ground_truth:
                correct_predictions += 1

            for p_char, g_char in zip(prediction, ground_truth):
                if p_char == g_char:
                    correct_chars += 1

            processed_count += 1
            total_chars_count += captcha_len

            if i % 100 == 0:
                print(f"Processed {i} images...")

    if processed_count > 0:
        accuracy_total = (correct_predictions / processed_count) * 100
        accuracy_char = (correct_chars / total_chars_count) * 100

        print("-" * 35)
        print("Testing Complete!")
        print(f"Samples Processed: {processed_count}")
        print(f"Total Characters: {total_chars_count}")
        print("-" * 35)
        print(f"Full String Matches: {correct_predictions}")
        print(f"Full String Accuracy: {accuracy_total:.2f}%")
        print("-" * 35)
        print(f"Character Matches: {correct_chars}")
        print(f"Character Accuracy: {accuracy_char:.2f}%")
        print("-" * 35)
    else:
        print("No valid files found for testing.")

calculate_accuracy(model, 5000)
#Usage for testing: Place the "data" folder in the same directory as this file. The "data" folder contains img1.png to img5000.png, and img1.txt to img5000.txt.