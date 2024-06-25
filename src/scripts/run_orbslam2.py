import os
import subprocess

# Define the sequences
sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
dataset_folder = "/media/adam/Data/ood_slam_data/datasets"  # This is the mounted folder inside the container
ORB_SLAM_FOLDER = "/home/adam/Documents/research_internship/ORB_SLAM2"

def run_orbslam2(sequence):

    if sequence in ["00", "01", "02"]:
        yaml_file = "Examples/Stereo/KITTI00-02.yaml"
    elif sequence == "03":
        yaml_file = "Examples/Stereo/KITTI03.yaml"
    else:
        yaml_file = "Examples/Stereo/KITTI04-12.yaml"

    command = (
        f"cd {ORB_SLAM_FOLDER} && ./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt {yaml_file} {dataset_folder}/kitti/odometry_gray/sequences/{sequence} "
        f" && mv CameraTrajectory.txt /home/adam/Documents/research_internship/OOD_SLAM/ood_slam/src/results/orbslam_poses/stereo/{sequence}.txt && exit"
    )

    subprocess.run(command, shell=True)

if __name__ == "__main__":
    for sequence in sequences:
        run_orbslam2(sequence)
