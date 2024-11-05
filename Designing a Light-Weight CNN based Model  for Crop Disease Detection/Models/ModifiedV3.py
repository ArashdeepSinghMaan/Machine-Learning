import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


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
        # Initial Convolution Layer
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


def create_dataloaders(data_dir):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True,num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return image_datasets,dataloaders,dataset_sizes,class_names

if __name__ == '__main__':
    data_dir = r"C:\Users\AMREEN\OneDrive - Indian Institute of Technology Jodhpur\Desktop\SEM 3\FML CSL7670\Project\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
    
   
    image_datasets,dataloaders,dataset_sizes,class_names = create_dataloaders(data_dir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_classes = len(class_names)
    model = SimpleMobileNetV3Lite(num_classes=num_classes).to(device)

  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    performance_data = {
         "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "confusion_matrix": [],
        "train_time_per_epoch": [],
        "inference_time_per_sample": [],
        "model_size_MB": 0,
        "num_parameters": 0
    }

  
    def train_model(model, criterion, optimizer, num_epochs=25):
        since = time.time()

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            epoch_start_time = time.time()  

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train() 
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                all_preds = []
                all_labels = []

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

              
                    if phase == 'val':
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    performance_data['train_loss'].append(epoch_loss)
                    performance_data['train_acc'].append(epoch_acc.item())
                else:
                    performance_data['val_loss'].append(epoch_loss)
                    performance_data['val_acc'].append(epoch_acc.item())

                  
                    precision = precision_score(all_labels, all_preds, average='weighted')
                    recall = recall_score(all_labels, all_preds, average='weighted')
                    f1 = f1_score(all_labels, all_preds, average='weighted')
                    conf_matrix = confusion_matrix(all_labels, all_preds)

                    
                    performance_data['precision'].append(precision)
                    performance_data['recall'].append(recall)
                    performance_data['f1_score'].append(f1)
                    performance_data['confusion_matrix'].append(conf_matrix.tolist())

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

  
            epoch_time = time.time() - epoch_start_time
            performance_data['train_time_per_epoch'].append(epoch_time)
            print(f'Epoch {epoch} took {epoch_time:.2f} seconds')

        total_time = time.time() - since
        print(f'Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')

        return model

    inference_times = []
    for i, (inputs, labels) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(inputs)
        inference_time = time.time() - start_time
        inference_times.append(inference_time / inputs.size(0))
    
    performance_data['inference_time_per_sample'] = np.mean(inference_times)
   
    model=train_model(model , criterion , optimizer , num_epochs=25)

    torch.save(model.state_dict(), 'trial_2.pth')
    
    print(f"Model saved as 'trial_2.pth'")
    model_size_MB = os.path.getsize('trial_2.pth') / (1024 * 1024)
    performance_data['model_size_MB'] = model_size_MB

 
    num_parameters = sum(p.numel() for p in model.parameters())
    performance_data['num_parameters'] = num_parameters

      
    with open('ModifiedMobileNet_model_performance_data.json', 'w') as f:
            json.dump(performance_data, f, indent=4)

    print("Model performance data and trained model saved.")
