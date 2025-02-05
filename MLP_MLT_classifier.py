import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import softmax

# Multitask MLP model
class mltMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes_task1, num_classes_task2, drop_out=0.2):
        super(mltMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)

        # Task 1 output layer
        self.fc_task1 = nn.Linear(hidden_size, num_classes_task1)

        # Task 2 output layer
        self.fc_task2 = nn.Linear(hidden_size, num_classes_task2)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)

        # Task 1
        out_task1 = self.fc_task1(out)

        # Task 2
        out_task2 = self.fc_task2(out)

        return out_task1, out_task2
    
def calculate_metrics(labels, predictions, probs):
    accuracy = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, probs[:, 1])
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return accuracy, auc, precision, recall, f1

seed = 2023
torch.manual_seed(seed)
np.random.seed(seed)

# Load protein sequence embeddings from CSV
data = pd.read_csv("mean_embeddings_esm2_t12.csv")

embeddings = data.iloc[:, 0:480].values 

labels1 = data['Label1']
labels2 = data['Label2']

# Encode your labels as integers
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
labels1 = label_encoder1.fit_transform(labels1)
labels2 = label_encoder2.fit_transform(labels2)

combined_labels = list(zip(labels1, labels2))

# Initialize your model
input_size = len(embeddings[0])
# print(input_size)
hidden_size = 128  
num_classes1 = len(label_encoder1.classes_)  # Number of unique classes
num_classes2 = len(label_encoder2.classes_)

l = 0.4

# Split the data into training, validation, and test sets with stratified sampling
X_train, X_temp, y_train_combined, y_temp_combined = train_test_split(embeddings, combined_labels, test_size=0.4, stratify=combined_labels, random_state = 1)
X_val, X_test, y_val_combined, y_test_combined = train_test_split(X_temp, y_temp_combined, test_size=0.5, stratify=y_temp_combined, random_state = 1)

# Separate the combined labels back into labels for each task
y_train1, y_train2 = zip(*y_train_combined)
y_val1, y_val2 = zip(*y_val_combined)
y_test1, y_test2 = zip(*y_test_combined)

# Inverse transform the labels to get the original class labels
y_train_labels1 = label_encoder1.inverse_transform(y_train1)
y_train_labels2 = label_encoder2.inverse_transform(y_train2)
y_val_labels1 = label_encoder1.inverse_transform(y_val1)
y_val_labels2 = label_encoder2.inverse_transform(y_val2)
y_test_labels1 = label_encoder1.inverse_transform(y_test1)
y_test_labels2 = label_encoder2.inverse_transform(y_test2)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train1 = torch.tensor(y_train1, dtype=torch.int64)
y_train2 = torch.tensor(y_train2, dtype=torch.int64)
y_val1 = torch.tensor(y_val1, dtype=torch.int64)
y_val2 = torch.tensor(y_val2, dtype=torch.int64)
y_test1 = torch.tensor(y_test1, dtype=torch.int64)
y_test2 = torch.tensor(y_test2, dtype=torch.int64)

# Create DataLoader for training, validation, and testing data
train_dataset = TensorDataset(X_train, y_train1, y_train2)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = TensorDataset(X_val, y_val1, y_val2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataset = TensorDataset(X_test, y_test1, y_test2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = mltMLP(input_size, hidden_size, num_classes1, num_classes2)

# Define loss functions for each task
criterion_task1 = nn.CrossEntropyLoss()
criterion_task2 = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize variables to keep track of the lowest validation loss and corresponding model weights
lowest_val_loss = float('inf')
best_model_weights = None

# Training loop
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    for inputs, labels_task1, labels_task2 in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs_task1, outputs_task2 = model(inputs)
        
        # Calculate losses for each task
        loss_task1 = criterion_task1(outputs_task1, labels_task1)
        loss_task2 = criterion_task2(outputs_task2, labels_task2)
        
        # Total loss is the sum of losses for both tasks
        total_loss = l * loss_task1 + (1 - l) *loss_task2
        
        # Backward and optimize
        total_loss.backward()
        optimizer.step()

    # Validation on the validation set to check for the lowest validation loss
    model.eval()
    with torch.no_grad():
        total_val_loss_task1 = 0.0
        total_val_loss_task2 = 0.0
        for inputs, labels_task1, labels_task2 in val_loader:
            outputs_task1, outputs_task2 = model(inputs)
            val_loss_task1 = criterion_task1(outputs_task1, labels_task1)
            val_loss_task2 = criterion_task2(outputs_task2, labels_task2)
            total_val_loss_task1 += val_loss_task1.item()
            total_val_loss_task2 += val_loss_task2.item()

        # Calculate the average validation loss for each task
        avg_val_loss_task1 = total_val_loss_task1 / len(val_loader)
        avg_val_loss_task2 = total_val_loss_task2 / len(val_loader)

        # Total validation loss is a weighted sum of losses for both tasks
        avg_val_loss = l * avg_val_loss_task1 + (1 - l) * avg_val_loss_task2

        # Check if the current model has a lower validation loss than the lowest recorded
        if avg_val_loss < lowest_val_loss:
            lowest_val_loss = avg_val_loss
            best_model_weights = model.state_dict()
            # Save the best model weights
            torch.save(best_model_weights, 'best_model_2_tasks.pth')

model.load_state_dict(torch.load('best_model_2_tasks.pth'))

# Evaluation
model.eval()
all_predictions_task1 = []
all_predictions_task2 = []
all_probs_task1 = []
all_probs_task2 = []
all_labels_task1 = []
all_labels_task2 = []

with torch.no_grad():
    for inputs, labels_task1, labels_task2 in test_loader:
        outputs_task1, outputs_task2 = model(inputs)

        # Get predicted classes
        _, predicted_task1 = torch.max(outputs_task1.data, 1)
        _, predicted_task2 = torch.max(outputs_task2.data, 1)

        # Store raw probabilities for AUC calculation
        probs_task1 = softmax(outputs_task1, dim=1).cpu().numpy()
        probs_task2 = softmax(outputs_task2, dim=1).cpu().numpy()

        # Store predictions and labels
        all_predictions_task1.extend(predicted_task1.tolist())
        all_predictions_task2.extend(predicted_task2.tolist())
        all_probs_task1.extend(probs_task1)  # Store softmax probabilities for task 1
        all_probs_task2.extend(probs_task2)  # Store softmax probabilities for task 2
        all_labels_task1.extend(labels_task1.tolist())
        all_labels_task2.extend(labels_task2.tolist())

# Calculate metrics for each task
metrics_task1 = calculate_metrics(all_labels_task1, all_predictions_task1, np.array(all_probs_task1))
metrics_task2 = calculate_metrics(all_labels_task2, all_predictions_task2, np.array(all_probs_task2))
    
print(f"Accuracy on the test set: Task 1 - {metrics_task1[0]:.2f}%, Task 2 - {metrics_task2[0]:.2f}%")


