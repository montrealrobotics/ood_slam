import os
import shutil
import random

def split_data(raw_data_dir, output_dir, train_ratio=0.8):
    sequences_dir = os.path.join(raw_data_dir, 'sequences')
    labels_dir = os.path.join(raw_data_dir, 'labels')
    
    train_img0_dir = os.path.join(output_dir, 'train', 'images', 'image_0')
    train_img1_dir = os.path.join(output_dir, 'train', 'images', 'image_1')
    train_lbl_dir = os.path.join(output_dir, 'train', 'labels')
    
    val_img0_dir = os.path.join(output_dir, 'val', 'images', 'image_0')
    val_img1_dir = os.path.join(output_dir, 'val', 'images', 'image_1')
    val_lbl_dir = os.path.join(output_dir, 'val', 'labels')
    
    os.makedirs(train_img0_dir, exist_ok=True)
    os.makedirs(train_img1_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img0_dir, exist_ok=True)
    os.makedirs(val_img1_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    
    sequences = [seq for seq in os.listdir(sequences_dir) if os.path.isdir(os.path.join(sequences_dir, seq))]
    

    for sequence in sequences:
        img0_dir = os.path.join(sequences_dir, sequence, 'image_0')
        img1_dir = os.path.join(sequences_dir, sequence, 'image_1')
        lbl_dir = os.path.join(labels_dir, sequence)
        
        img_files = sorted([f for f in os.listdir(img0_dir)])
        
        random.shuffle(img_files)
        
        train_split = int(train_ratio * len(img_files))
        
        for i, img_file in enumerate(img_files):
            img0_path = os.path.join(img0_dir, img_file)
            img1_path = os.path.join(img1_dir, img_file)
            lbl_path = os.path.join(lbl_dir, f'rpe_{img_file.split(".")[0]}.txt')
            
            if i < train_split:
                shutil.copy(img0_path, os.path.join(train_img0_dir, sequence + '_' + img_file))
                shutil.copy(img1_path, os.path.join(train_img1_dir, sequence + '_' + img_file))
                shutil.copy(lbl_path, os.path.join(train_lbl_dir, sequence + '_' + f'rpe_{img_file.split(".")[0]}.txt'))
            else:
                shutil.copy(img0_path, os.path.join(val_img0_dir, sequence + '_' + img_file))
                shutil.copy(img1_path, os.path.join(val_img1_dir, sequence + '_' + img_file))
                shutil.copy(lbl_path, os.path.join(val_lbl_dir, sequence + '_' + f'rpe_{img_file.split(".")[0]}.txt'))

if __name__ == "__main__":
    raw_data_dir = '/media/adam/T9/ood_slam_data/datasets/kitti/odometry_gray'
    output_dir = '/media/adam/T9/ood_slam_data/datasets/pytorch_data/orb_slam/stereo'
    split_data(raw_data_dir, output_dir, train_ratio=0.8)
