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

print("Please place this file and the png file, CNN_model.pth, in the same directory, then enter the name of the png file.")
print("Output @ to exit the program")
while 1 :
    path = input("").strip()
    if path == '@':
        break
    script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(script_path)
    path1 = os.path.join(current_dir, path)
    if not os.path.exists(path1):
        print("Incorrect name")
        continue
    result = predict(model, path1)
    print(f"result:{result}")