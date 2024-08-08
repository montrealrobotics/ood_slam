import sys
import os

# Add the project root to the Python path before importing any modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from models.alexNetRegression import AlexNetSLAMRegressor
import torch.nn.functional as F
from utils.dataloader import get_dataloaders
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0.0
    model.train()

    for data, labels in iterator:
        left_images, right_images = data
        images = torch.cat((left_images, right_images), dim=1).to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)


        # Backward pass
        loss.backward()

        # Update model params
        optimizer.step()

        # Accumulate model params
        epoch_loss += loss.item() * images.size(0)

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0.0
    model.eval()

    with torch.no_grad():
        for data, labels in iterator:
            left_images, right_images = data
            images = torch.cat((left_images, right_images), dim=1).to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            epoch_loss += loss.item() * images.size(0)

    return epoch_loss / len(iterator)

def calculate_r_squared(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item() 

def eval_r_squared(model, iterator, device):
    model.eval()  # Set the model to evaluation mode
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, targets in iterator:
            left_images, right_images = inputs
            images = torch.cat((left_images, right_images), dim=1).to(device)
            targets = targets.to(device)  # Move to the appropriate device
            outputs = model(images)
            y_true_list.append(targets)
            y_pred_list.append(outputs)

    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)

    return calculate_r_squared(y_true, y_pred)


def create_aggregated_probability_matrix(model, dataloader, num_classes):
    """
    Create a matrix that aggregates the predicted probability distributions 
    for each true label across all samples in the validation dataset.

    Parameters:
    - model: The trained PyTorch model.
    - dataloader: The DataLoader for the validation dataset.
    - num_classes: The number of classes.

    Returns:
    - agg_prob_matrix: A 2D numpy array of shape (num_classes, num_classes) 
                       representing the aggregated probability distributions.
    """

    model.eval()

    agg_prob_matrix_rotation = np.zeros((num_classes, num_classes))
    agg_prob_matrix_translation = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for data, labels in dataloader:
            left_images, right_images = data
            images = torch.cat((left_images, right_images), dim=1).to(device)

            # Forward pass to get outputs
            outputs1, outputs2 = model(images)


            # Convert outputs to probability distributions
            prob_dist1 = torch.exp(torch.log_softmax(outputs1, dim=1))
            prob_dist2 = torch.exp(torch.log_softmax(outputs2, dim=1))


            for i in range(prob_dist1.size(0)):
                true_label = labels[i, 0].item()
                agg_prob_matrix_rotation[true_label] += prob_dist1[i].cpu().numpy()

            for i in range(prob_dist2.size(0)):
                true_label = labels[i, 1].item()
                agg_prob_matrix_translation[true_label] += prob_dist2[i].cpu().numpy()
    
    return agg_prob_matrix_rotation, agg_prob_matrix_translation

def visualize_confusion_matrix(matrix, component, output_dir):
    # Normalize matrix
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = matrix / row_sums

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=np.arange(len(matrix)), yticklabels=np.arange(len(matrix)),
                vmin=0, vmax=1)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"Aggregated Probability Distribution Matrix ({component})")

    plt.savefig(f"{output_dir}/aggregated_probability_matrix_{component}.png")

    plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize wandb
    wandb.init(project="ood-slam", entity="udem-mila", mode="offline", name="alexnet_regression")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = config['dataset']['train_data_dir']
    val_dir = config['dataset']['val_data_dir']
    batch_size = config['dataset']['batch_size']
    sequence_length = config['dataset']['sequence_length']
    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']


    train_loader, val_loader = get_dataloaders(train_dir, val_dir, batch_size, sequence_length, train_transforms, val_transforms)

    model = AlexNetSLAMRegressor(config['model']['weights_path'])

    output_dir = config['model']['output_dir']
    
    criterion = nn.MSELoss()
    params = (param for param in model.parameters() if param.requires_grad)
    optimizer = optim.Adam(params, lr=1e-5)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    

    model = model.to(device)

    # Log hyperparameters
    wandb.config.update({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "sequence_length": sequence_length
    })

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {train_loss}")
        wandb.log({"train_loss": train_loss, "epoch": epoch + 1})

        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Val loss: {val_loss}")
        wandb.log({"val_loss": val_loss, "epoch": epoch + 1})

    output_dir = config['model']['output_dir']
    torch.save(model.state_dict(), f"{output_dir}/fine_tuned_alexnet_weights.pth")

    r_squared = eval_r_squared(model, val_loader, device)
    print(f'R-squared: {r_squared:.4f}')

    # mat1, mat2 = create_aggregated_probability_matrix(model, val_loader, 5)

    # visualize_confusion_matrix(mat1, "rotation", output_dir)
    # visualize_confusion_matrix(mat2, "translation", output_dir)