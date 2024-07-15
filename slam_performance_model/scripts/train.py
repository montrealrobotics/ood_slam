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
from models.alexNetRegression import AlexNetSLAM
from utils.dataloader import get_dataloaders
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train_model(config):
# Initialize wandb
    wandb.init(project="ood-slam", entity="udem-mila", mode="offline")

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

    train_loader, val_loader = get_dataloaders(train_dir, val_dir, batch_size, sequence_length, train_transforms, val_transforms)

    model = AlexNetSLAM(config['model']['weights_path'], num_classes=2)
    model = model.to(device)

    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Log hyperparameters
    wandb.config.update({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "sequence_length": sequence_length
    })

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, labels in train_loader:
            left_images, right_images = data
            images = torch.cat((left_images, right_images), dim=1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
        wandb.log({"train_loss": epoch_loss, "epoch": epoch + 1})

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                left_images, right_images = data
                images = torch.cat((left_images, right_images), dim=1).to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {epoch_val_loss}")
        wandb.log({"val_loss": epoch_val_loss, "epoch": epoch + 1})
    
    # Save model
    torch.save(model.state_dict(), "model.pth")
    wandb.save("model.pth")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train_model(config)