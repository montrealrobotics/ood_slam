import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from model import AlexNetSLAM
from dataloader import StereoSLAMDataset, get_dataloaders

if __name__ == "__main__":
    data_dir = '/media/adam/Data/ood_slam_data/datasets/pytorch_data/orb_slam/stereo'
    batch_size = 32
    sequence_length = 1
    num_epochs = 25
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader, val_loader = get_dataloaders(data_dir, batch_size, sequence_length, train_transforms, val_transforms)

    model = AlexNetSLAM(num_classes=2)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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