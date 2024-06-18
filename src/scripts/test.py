import numpy as np
import matplotlib.pyplot as plt

def load_ground_truth(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            R = np.array(values[:9]).reshape(3, 3)
            t = np.array(values[9:12]).reshape(3, 1)
            poses.append((R, t))
    return poses

def load_estimated(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            t = np.array(values[:3])
            q = np.array(values[3:])
            poses.append((t, q))
    return poses

def plot_trajectories(gt_poses, est_poses):
    gt_x = [t[0, 0] for _, t in gt_poses]
    gt_y = [t[1, 0] for _, t in gt_poses]
    
    est_x = [t[0] for t, _ in est_poses]
    est_y = [t[1] for t, _ in est_poses]

    plt.figure()
    plt.plot(gt_x, gt_y, label='Ground Truth')
    plt.plot(est_x, est_y, label='Estimated', linestyle='--')
    plt.legend()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory Comparison')
    plt.show()

def main():
    # File paths for ground truth and estimated poses
    gt_file_path = '/home/adam/Documents/research_internship/OOD_SLAM/src/results/ground_truth_poses/00.txt'
    est_file_path = '/home/adam/Documents/research_internship/OOD_SLAM/src/results/floam_poses/data/00.txt'
    
    
    # Load poses
    gt_poses = load_ground_truth(gt_file_path)
    est_poses = load_estimated(est_file_path)

    print(gt_poses)
    
    # Plot trajectories
    plot_trajectories(gt_poses, est_poses)

if __name__ == '__main__':
    main()


