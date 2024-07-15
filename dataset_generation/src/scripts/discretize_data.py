import os
import numpy as np
import pandas as pd

def create_subfolders(data_dir):
    regression_dir = os.path.join(data_dir, 'labels', 'regression')
    classification_dir = os.path.join(data_dir, 'labels', 'classification')
    os.makedirs(regression_dir, exist_ok=True)
    os.makedirs(classification_dir, exist_ok=True)

def load_labels(label_dir, num_sequences):
    labels = []
    for seq in range(num_sequences):
        seq_labels = []
        for file in sorted(os.listdir(label_dir)):
            if file.startswith(f"{seq:02d}_rpe_") and file.endswith(".txt"):
                label_path = os.path.join(label_dir, file)
                label = np.loadtxt(label_path, dtype=np.float32)
                seq_labels.append(label)
        labels.extend(seq_labels)
    return np.array(labels)

def manual_qcut(values, n_quantiles):
    # First we have to sort the values
    sorted_indices = np.argsort(values)

    # Then we determine the number of values per quantile
    n_values = len(values)
    values_per_quantile = n_values // n_quantiles
    remainder = n_values % n_quantiles

    # Assign quantiles
    quantiles = np.zeros(n_values, dtype=int)
    current_index = 0
    for i in range(n_quantiles):
        quantile_size = values_per_quantile + (1 if i < remainder else 0)
        quantiles[sorted_indices[current_index : current_index + quantile_size]] = i
        current_index += quantile_size
    
    return quantiles

def discretize_labels(labels, n_quantiles):
    rotational_errors = labels[:, 0]
    translational_errors = labels[:, 1]
    # Discretize translational and rotational errors into quantiles
    translational_quantiles = manual_qcut(translational_errors, n_quantiles)
    rotational_quantiles = manual_qcut(rotational_errors, n_quantiles)
    
    discretized_labels = np.column_stack((translational_quantiles, rotational_quantiles))
    return discretized_labels

def save_discretized_labels(discretized_labels, original_labels, label_dir, num_sequences):
    regression_dir = os.path.join(label_dir, 'regression')
    classification_dir = os.path.join(label_dir, 'classification')
    
    idx = 0
    for seq in range(num_sequences):
        for file in sorted(os.listdir(label_dir)):
            if file.startswith(f"{seq:02d}_rpe_") and file.endswith(".txt"):
                # Save original regression labels
                original_label_path = os.path.join(regression_dir, file)
                np.savetxt(original_label_path, original_labels[idx], fmt='%.6f')
                
                # Save discretized classification labels
                classification_label_path = os.path.join(classification_dir, file)
                np.savetxt(classification_label_path, discretized_labels[idx], fmt='%d')
                
                idx += 1

def process_data(data_dir, num_sequences, n_quantiles):
    label_dir = os.path.join(data_dir, 'labels')
    create_subfolders(data_dir)
    
    original_labels = load_labels(label_dir, num_sequences)
    discretized_labels = discretize_labels(original_labels, n_quantiles)
    save_discretized_labels(discretized_labels, original_labels, label_dir, num_sequences)

if __name__ == "__main__":

    train_data_dir = '/media/adam/T9/ood_slam_data/datasets/pytorch_data/orb_slam/stereo/train'
    val_data_dir = '/media/adam/T9/ood_slam_data/datasets/pytorch_data/orb_slam/stereo/val'
    num_sequences = 11  # Assuming you have 11 sequences
    n_quantiles = 5  # Number of quantiles to discretize into

    # Process train and val directories
    process_data(train_data_dir, num_sequences, n_quantiles)
    process_data(val_data_dir, num_sequences, n_quantiles)
