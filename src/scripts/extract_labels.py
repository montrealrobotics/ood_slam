import os
import numpy as np

def extract_labels(errors_dir, raw_data_dir, output_dir):
    sequences_dir = os.path.join(raw_data_dir, 'sequences')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for sequence in os.listdir(sequences_dir):
        seq_dir = os.path.join(sequences_dir, sequence)
        if not os.path.isdir(seq_dir):
            continue
        
        label_dir = os.path.join(output_dir, sequence)
        os.makedirs(label_dir, exist_ok=True)
        
        rpe_file_path = os.path.join(errors_dir, f'{sequence}.txt')
        if not os.path.exists(rpe_file_path):
            print(f"RPE file {rpe_file_path} not found. Skipping sequence {sequence}.")
            continue
        
        rpe_data = np.loadtxt(rpe_file_path)

        for i in range(-1, len(rpe_data)):
            if i == -1:
                translational_error = 0.0
                rotational_error = 0.0
            else:
                translational_error = rpe_data[i, 1]
                rotational_error = rpe_data[i, 2]
            
            label = np.array([translational_error, rotational_error])
            
            label_file_path = os.path.join(label_dir, f'rpe_{i+1:06d}.txt')
            np.savetxt(label_file_path, label)
                    
        print(f"Labels for sequence {sequence} extracted and saved.")

if __name__ == "__main__":
    errors_dir = "/home/adam/Documents/research_internship/OOD_SLAM/ood_slam/src/results/orbslam_poses/stereo/errors"
    raw_data_dir = '/media/adam/Data/ood_slam_data/datasets/kitti/odometry_gray'
    output_dir = os.path.join(raw_data_dir, 'labels')
    extract_labels(errors_dir, raw_data_dir, output_dir)
