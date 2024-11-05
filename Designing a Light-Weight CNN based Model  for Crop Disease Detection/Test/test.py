import torch
from torchvision import datasets, transforms, models
from PIL import Image
import os

# Load the model with the same architecture used during training
model = models.mobilenet_v2(weights=None)  # No pre-trained weights
num_classes = 38  # Update this if your dataset has a different number of classes
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

# Load the saved model weights
model.load_state_dict(torch.load('mobilenetv2_plant_disease.pth'))

# Set model to evaluation mode (important for inference)
model.eval()

# Move the model to the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the transformation for the test image (must match what was used during training)
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset to get class names (or use a saved list of class names)
data_dir = r"C:\Users\AMREEN\OneDrive - Indian Institute of Technology Jodhpur\Desktop\SEM 3\FML CSL7670\Project\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['train', 'val']}
class_names = image_datasets['train'].classes  # List of class names from training data

# Function to load and preprocess an image
def process_image(image_path):
    image = Image.open(image_path)
    image = data_transforms(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to make a prediction on a single image
def predict_image(image_path, model, class_names):
    image = process_image(image_path).to(device)
    with torch.no_grad():  # Inference doesn't need gradients
        outputs = model(image)
        _, preds = torch.max(outputs, 1)  # Get the index of the highest score
        predicted_class = class_names[preds[0]]
    return predicted_class

# Path to the image you want to test
test_image_path = r"C:\Users\AMREEN\OneDrive - Indian Institute of Technology Jodhpur\Desktop\SEM 3\FML CSL7670\Project\archive\test\test\TomatoYellowCurlVirus4.JPG"
 # Replace with the actual path to your test image

# Make prediction
predicted_class = predict_image(test_image_path, model, class_names)

print(f'The model predicted: {predicted_class}')


