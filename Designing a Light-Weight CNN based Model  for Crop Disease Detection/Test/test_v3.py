import torch
from torchvision import datasets, transforms, models
from PIL import Image
import os
import time
import matplotlib.pyplot as plt

# Load the model with the same architecture used during training
model = models.mobilenet_v2(weights=None)  # No pre-trained weights
num_classes = 38  # Update this if your dataset has a different number of classes
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

# Load the saved model weights
model.load_state_dict(torch.load('mobilenetv3_GSD_small_plant_disease.pth'))

# Set model to evaluation mode (important for inference)
model.eval()

# Move the model to the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the transformation for the test images
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

# Function to make predictions on a folder of images
def predict_images(image_folder, model, class_names):
    results = []  # Store results for each image

    # Iterate over all images in the folder
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image file types
            image = process_image(img_path).to(device)

            # Measure inference time
            start_time = time.time()
            with torch.no_grad():  # Inference doesn't need gradients
                outputs = model(image)
                probs = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities
                _, preds = torch.max(outputs, 1)  # Get the index of the highest score
            end_time = time.time()

            predicted_class = class_names[preds[0]]
            confidence = probs[0][preds[0]].item()
            inference_time = end_time - start_time

            results.append({
                'image': img_file,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'inference_time': inference_time
            })

            # Display the image with the predicted class and confidence
            plt.imshow(Image.open(img_path))
            plt.title(f'Predicted: {predicted_class} ({confidence:.2f})')
            plt.axis('off')
            plt.show()

    return results

# Path to the folder containing test images
test_image_folder = r"C:\Users\AMREEN\OneDrive - Indian Institute of Technology Jodhpur\Desktop\SEM 3\FML CSL7670\Project\archive\test\test"

# Make predictions
results = predict_images(test_image_folder, model, class_names)

# Save results to a file
output_file = 'model_predictions.txt'
with open(output_file, 'w') as f:
    for result in results:
        f.write(f"Image: {result['image']}, Predicted Class: {result['predicted_class']}, "
                f"Confidence: {result['confidence']:.4f}, Inference Time: {result['inference_time']:.4f} seconds\n") 
print(f"Results saved to {output_file}")
