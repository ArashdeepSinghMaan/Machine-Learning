import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 1st convolution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 2nd convolution layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # Fully connected layer
        self.fc2 = nn.Linear(512, num_classes)  # Output layer

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return image_datasets, dataloaders, dataset_sizes, class_names

if __name__ == '__main__':
    data_dir = r"C:\Users\AMREEN\OneDrive - Indian Institute of Technology Jodhpur\Desktop\SEM 3\FML CSL7670\Project\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"  # Update with your data directory
    

    image_datasets, dataloaders, dataset_sizes, class_names = create_dataloaders(data_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

 
    num_classes = len(class_names)
    model = SimpleCNN(num_classes=num_classes).to(device)

 
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


   
    model = train_model(model, criterion, optimizer, num_epochs=25)

       
    num_parameters = sum(p.numel() for p in model.parameters())
    performance_data['num_parameters'] = num_parameters

  
    inference_times = []
    for i, (inputs, labels) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(inputs)
        inference_time = time.time() - start_time
        inference_times.append(inference_time / inputs.size(0))

    performance_data['inference_time_per_sample'] = np.mean(inference_times)

      
    torch.save(model.state_dict(), 'simple_cnn_model.pth')

    print(f"Model with {num_parameters} parameters saved as 'simple_cnn_model.pth'")
        
    model_size_MB = os.path.getsize('simple_cnn_model.pth') / (1024 * 1024)
    performance_data['model_size_MB'] = model_size_MB

      
    with open('simpleCNN_model_performance_data.json', 'w') as f:
            json.dump(performance_data, f, indent=4)

    print("Model performance data and trained model saved.")
