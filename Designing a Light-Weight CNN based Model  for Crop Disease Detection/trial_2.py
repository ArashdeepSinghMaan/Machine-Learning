import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import torch.nn.functional as F

# Hard-Swish activation function
class HardSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

# Squeeze-and-Excitation block
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

# Bottleneck Block with optional SE block
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, kernel_size, use_se=False, activation=nn.ReLU):
        super(BottleneckBlock, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.use_se = use_se
        self.block = nn.Sequential(
            # 1x1 pointwise conv (expand)
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(),
            # Depthwise conv
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(),
            # Optional SE block
            SqueezeExcitation(hidden_dim) if use_se else nn.Identity(),
            # 1x1 pointwise conv (project)
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

# Define the modified SimpleMobileNetV3Lite architecture
class SimpleMobileNetV3Lite(nn.Module):
    def __init__(self, num_classes):
        super(SimpleMobileNetV3Lite, self).__init__()
        # Initial Convolution Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )
        # Bottleneck blocks with SE and HardSwish where appropriate
        self.bottlenecks = nn.Sequential(
            BottleneckBlock(16, 16, expansion_factor=1, kernel_size=3, use_se=True),  # Initial bottleneck
            BottleneckBlock(16, 24, expansion_factor=4, kernel_size=3),  # Without SE
            BottleneckBlock(24, 24, expansion_factor=4, kernel_size=3),  # Without SE
            BottleneckBlock(24, 40, expansion_factor=4, kernel_size=5, use_se=True),  # With SE
            BottleneckBlock(40, 40, expansion_factor=4, kernel_size=5, use_se=True)   # With SE
        )
        # Final convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            HardSwish()
        )
        # Pooling and fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottlenecks(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)
        return x

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create datasets and dataloaders
def create_dataloaders(data_dir):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True,num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return image_datasets,dataloaders,dataset_sizes,class_names

if __name__ == '__main__':
    data_dir = r"C:\Users\AMREEN\OneDrive - Indian Institute of Technology Jodhpur\Desktop\SEM 3\FML CSL7670\Project\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
    
    # Create datasets and dataloaders
    image_datasets,dataloaders,dataset_sizes,class_names = create_dataloaders(data_dir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the modified CNN model
    num_classes = len(class_names)
    model = SimpleMobileNetV3Lite(num_classes=num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training function
    def train_model(model , criterion , optimizer , num_epochs=25):
        since=time.time()
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train() 
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs , labels in dataloaders[phase]:
                    inputs ,labels = inputs.to(device) , labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs=model(inputs)
                        _, preds=torch.max(outputs , 1)
                        loss=criterion(outputs , labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        return model

    # Train the modified CNN model
    model=train_model(model , criterion , optimizer , num_epochs=25)

    torch.save(model.state_dict(), 'trial_2.pth')
    
    print(f"Model saved as 'trial_2.pth'")