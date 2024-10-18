import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import random

class KITTIDataset(Dataset):
    def __init__(self, sequence_dirs, flipped_sequence_dirs, label_files, sequence_length=2, transform=None):
        """
        Args:
            sequence_dirs (list): List of directories with images for each KITTI sequence.
            label_files (list): List of CSV files with RPE labels for each sequence.
            sequence_length (int): Number of consecutive frames in a sequence (default is 2).
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.sequence_dirs = sequence_dirs
        self.flipped_sequence_dirs = flipped_sequence_dirs
        self.sequence_length = sequence_length
        self.transform = transform

        # Load all label files into a list of DataFrames, one per sequence
        self.labels = [pd.read_csv(label_file) for label_file in label_files]

        # Calculate and store the length of each sequence (minus sequence_length + 1)
        self.seq_lengths = [len(label_df) - sequence_length + 1 for label_df in self.labels]

        # Store cumulative lengths for indexing across multiple sequences
        self.cum_lengths = [sum(self.seq_lengths[:i+1]) for i in range(len(self.seq_lengths))]

    def __len__(self):
        # Total length is the sum of all sequence lengths
        return sum(self.seq_lengths)

    def _get_sequence_idx(self, idx):
        # Find which sequence idx falls into
        for seq_num, cum_len in enumerate(self.cum_lengths):
            if idx < cum_len:
                if seq_num == 0:
                    return seq_num, idx
                return seq_num, idx - self.cum_lengths[seq_num - 1]

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sequence to retrieve.
        
        Returns:
            A dictionary containing:
            - 'images': A tensor of shape [sequence_length, 6, H, W] where 6 is for concatenated RGB images.
            - 'rpe': A tensor containing the RPE (rotation_error, translation_error) for the last frame in the sequence.
        """
        # Determine which sequence and the specific frame within that sequence
        seq_num, frame_idx = self._get_sequence_idx(idx)

        # Randomly decide whether to use flipped data or not
        use_flipped = random.choice([True, False])

        base_dir = self.flipped_sequence_dirs if use_flipped else self.sequence_dirs

        frames = []  # To store the sequence of frames
        label_df = self.labels[seq_num]

        # Loop over sequence length to get consecutive frames
        for i in range(self.sequence_length):
            frame_id = str(int(label_df.iloc[frame_idx + i]['frame_id'])).zfill(6)
            seq_dir = base_dir[seq_num]  # Get the image directory for the current sequence

            # Load stereo images (left and right) from image_0 and image_1 directories
            img_left_path = os.path.join(seq_dir, "image_0", f"{frame_id}.png")
            img_right_path = os.path.join(seq_dir, "image_1", f"{frame_id}.png")
            img_left = Image.open(img_left_path).convert('RGB')
            img_right = Image.open(img_right_path).convert('RGB')

            # Apply transformations if any
            if self.transform:
                img_left = self.transform(img_left)
                img_right = self.transform(img_right)

            # Concatenate the left and right images along the channel dimension (6 channels: RGB for each)
            stereo_img = torch.cat((img_left, img_right), dim=0)
            frames.append(stereo_img)

        # Concatenate the frames to create a tensor of shape [sequence_length * channels, H, W]
        sequence = torch.cat(frames, dim=0)

        # Get the RPE (rotation and translation errors) for the last frame in the sequence
        rpe = label_df.iloc[frame_idx + self.sequence_length - 1][['rotation_quantile', 'translation_quantile']].values
        rpe = torch.tensor(rpe, dtype=torch.float32)
        rpe = rpe.type(torch.LongTensor)

        return {'images': sequence, 'rpe': rpe}


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    # Combine all the training sequences (00–08)
    train_sequence_dirs = [f'/media/adam/T9/ood_slam_data/datasets/kitti/odometry_gray/sequences/{i:02d}' for i in range(9)]
    train_label_files = [f'/media/adam/T9/slam_performance_model/data/errors/discretized/{i:02d}.csv' for i in range(9)]

    # Initialize the dataset
    train_dataset = KITTIDataset(sequence_dirs=train_sequence_dirs, label_files=train_label_files, sequence_length=2, transform=transform)

    # Create a DataLoader for the dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Test loading a batch of data
    for batch in train_loader:
        images = batch['images']  # Tensor of shape [batch_size, sequence_length, 6, H, W]
        rpe = batch['rpe']        # Tensor of shape [batch_size, 2]
        print(f"Images shape: {images.shape}, RPE shape: {rpe.shape}")
        break
