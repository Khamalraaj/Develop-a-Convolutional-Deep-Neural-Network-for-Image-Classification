# Develop a Convolutional Deep Neural Network for Image Classification


## AIM:
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET:
Include the Problem Statement and Dataset.

## Neural Network Model:
Include the neural network model diagram.

## DESIGN STEPS:
### STEP 1: 

Load the image dataset and divide it into training and testing datasets. Apply suitable transformations such as resizing and normalization to the images.

### STEP 2: 

Create DataLoader objects for the training and testing datasets to load images in batches.

### STEP 3: 

Design a CNN model consisting of convolution, ReLU activation, max-pooling, flattening, and fully connected layers.

### STEP 4: 

Initialize the CNN model, cross-entropy loss function, and Adam optimizer. Move the model to the available CPU/GPU device.

### STEP 5: 

Train the CNN for the specified number of epochs by performing forward propagation, calculating loss, backpropagation, and updating the model parameters.

### STEP 6: 

Evaluate the trained model using test images. Calculate accuracy, generate a confusion matrix and classification report, and verify the prediction using a new sample image.

### Name:Khamalraaj S

### Register Number:212224230122


## PROGRAM:

```

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Load Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)


# Get the shape of the first image in the training dataset
image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))

# Get the shape of the first image in the test dataset
image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))


# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # Input (1, 28, 28) -> Output (16, 28, 28)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # Output (16, 14, 14)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # Input (16, 14, 14) -> Output (32, 14, 14)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # Output (32, 7, 7)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128) # Flattened input: 32*7*7 = 1568 features
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)         # Output 10 classes for Fashion-MNIST

    def forward(self, x):
        # Apply first conv, relu, and pool
        x = self.pool1(self.relu1(self.conv1(x)))
        # Apply second conv, relu, and pool
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 32 * 7 * 7)

        # Apply fully connected layers
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x

from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name:khamalraaj S')
print('Register Number: 212224230122')
summary(model, input_size=(1, 28, 28))



# Initialize model, loss function, and optimizer
model = CNNClassifier()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name:khamalraaj S')
        print('Register Number: 212224230122')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Train the model
train_model(model, train_loader)

## Step 4: Test the Model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name:Khamalraaj S')
    print('Register Number:212224230122')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name:Khamalraaj S')
    print('Register Number:212224230122')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name:Khamalraaj S')
    print('Register Number:212224230122')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))


# Evaluate the model
# Redefining test_model to ensure images and labels are moved to the correct device
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device) # Move images to the device
            labels = labels.to(device) # Move labels to the device

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name:Khamalraaj S')
    print('Register Number:212224230122')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name:Khamalraaj S')
    print('Register Number:212224230122')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name:Khamalraaj S')
    print('Register Number:212224230122')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

test_model(model, test_loader)

## Step 5: Predict on a Single Image
import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes

    # Display the image
    print('Name:        ')
    print('Register Number:       ')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')

# Example Prediction
# Redefining predict_image to ensure the image is moved to the correct device
import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    # Move the image to the same device as the model
    image_to_predict = image.to(device)
    with torch.no_grad():
        output = model(image_to_predict.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)

    # Move the original image back to CPU for matplotlib to display if it was on GPU
    display_image = image.cpu() # ensure image is on CPU for plotting

    class_names = dataset.classes

    # Display the image
    print('Name:Khamalraaj S')
    print('Register Number: 212224230122')
    plt.imshow(display_image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')

predict_image(model, image_index=80, dataset=test_dataset)
```




### OUTPUT:

## Model  Summary

<img width="639" height="469" alt="image" src="https://github.com/user-attachments/assets/6e8dcbf6-e660-4032-83e5-641d856feae1" />

## Confusion Matrix

<img width="898" height="780" alt="image" src="https://github.com/user-attachments/assets/2349118b-e874-4db7-8a6f-c5711ec5b6f0" />


## Classification Report


<img width="550" height="408" alt="image" src="https://github.com/user-attachments/assets/b31eab62-fde7-48b0-9321-75e03a650aa5" />


### New Sample Data Prediction


<img width="507" height="559" alt="image" src="https://github.com/user-attachments/assets/7f53a65c-67a6-42b7-8ae7-2e274b491063" />


## RESULT:
The Convolutional Neural Network was successfully developed and trained for image classification using PyTorch.

