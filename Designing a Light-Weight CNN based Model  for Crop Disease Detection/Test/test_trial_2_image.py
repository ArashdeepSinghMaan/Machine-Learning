import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the same model architecture used during training
class HardSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, kernel_size, use_se=False, activation=nn.ReLU):
        super(BottleneckBlock, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.use_se = use_se
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(),
            SqueezeExcitation(hidden_dim) if use_se else nn.Identity(),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class SimpleMobileNetV3Lite(nn.Module):
    def __init__(self, num_classes):
        super(SimpleMobileNetV3Lite, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )
        self.bottlenecks = nn.Sequential(
            BottleneckBlock(16, 16, expansion_factor=1, kernel_size=3, use_se=True),
            BottleneckBlock(16, 24, expansion_factor=4, kernel_size=3),
            BottleneckBlock(24, 24, expansion_factor=4, kernel_size=3),
            BottleneckBlock(24, 40, expansion_factor=4, kernel_size=5, use_se=True),
            BottleneckBlock(40, 40, expansion_factor=4, kernel_size=5, use_se=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            HardSwish()
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottlenecks(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load the saved model weights
num_classes = 38  # Update if your dataset has a different number of classes
model = SimpleMobileNetV3Lite(num_classes=num_classes)
model.load_state_dict(torch.load('ModifiedV3.pth'))
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define transformations for the test image
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load class names from the dataset
data_dir =  r"C:\Users\AMREEN\OneDrive - Indian Institute of Technology Jodhpur\Desktop\SEM 3\FML CSL7670\Project\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Function to load and preprocess an image
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image)
    image = image.unsqueeze(0)
    return image

# Function to make a prediction and display the result
def display_prediction(image_path):
    image = process_image(image_path).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs.data.cpu(), 1)
        predicted_class = class_names[preds[0]]
    
    # Load the original image for display
    original_image = Image.open(image_path).convert('RGB')
    
    # Display the image with prediction text
    plt.figure(figsize=(6, 6))
    plt.imshow(original_image)
    plt.axis('off')
    plt.title(f"Prediction: {predicted_class}", fontsize=15, color='blue')
    plt.show()

# Test image path
test_image_path = r"C:\Users\AMREEN\OneDrive - Indian Institute of Technology Jodhpur\Desktop\SEM 3\FML CSL7670\Project\archive\test\test\AppleScab3.JPG" 
display_prediction(test_image_path)
