import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

class StereoSLAMDataset(Dataset):
    def __init__(self, data_dir, sequence_length=1, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images and labels.
            sequence_length (int): Number of frames in a sequence.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform

        self.image_paths_0 = []
        self.image_paths_1 = []
        self.label_paths = []

        img0_dir = os.path.join(data_dir, 'images', 'image_0')
        img1_dir = os.path.join(data_dir, 'images', 'image_1')
        label_dir = os.path.join(data_dir, 'labels', 'regression')

        img_files = sorted([f for f in os.listdir(img0_dir) if f.endswith('.png')])

        for img_file in img_files:
            img_0_path = os.path.join(img0_dir, img_file)
            img_1_path = os.path.join(img1_dir, img_file)
            label_file = img_file.replace('.png', '.txt').replace('_', '_rpe_')
            label_path = os.path.join(label_dir, label_file)

            self.image_paths_0.append(img_0_path)
            self.image_paths_1.append(img_1_path)
            self.label_paths.append(label_path)
    

    def __len__(self):
        return len(self.image_paths_0) - self.sequence_length + 1
    

    def __getitem__(self, idx):
        left_images = []
        right_images = []

        for i in range(self.sequence_length):
            left_img_path = self.image_paths_0[idx + i]
            right_img_path = self.image_paths_1[idx + i]

            left_image = Image.open(left_img_path).convert('L')  # 'L' mode for grayscale
            right_image = Image.open(right_img_path).convert('L')
            
            # Convert grayscale to RGB by duplicating channels
            left_image = left_image.convert('RGB')
            right_image = right_image.convert('RGB')


            if self.transform:
                left_image = self.transform(left_image)
                right_image = self.transform(right_image)

            left_images.append(left_image)
            right_images.append(right_image)

        # The label is associated with the last image of the sequence
        label_path = self.label_paths[idx + self.sequence_length - 1]
        label = np.loadtxt(label_path, dtype=np.float32)

        left_images = torch.stack(left_images) if self.sequence_length > 1 else left_images[0]
        right_images = torch.stack(right_images) if self.sequence_length > 1 else right_images[0]
        label = torch.tensor(label, dtype=torch.float32)

        return (left_images, right_images), label
    
def get_dataloaders(train_dir, val_dir, batch_size, sequence_length, train_transforms=None, val_transforms=None, num_workers=4):    
    train_dataset = StereoSLAMDataset(train_dir, sequence_length, transform=train_transforms)
    val_dataset = StereoSLAMDataset(val_dir, sequence_length, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
