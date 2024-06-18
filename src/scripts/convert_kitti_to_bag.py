import os
import subprocess
import re



def convert_kitti_to_bag(kitti_dir, output_dir, dates):
    for date in dates:
        date_dir = os.path.join(kitti_dir, date)
        
        if not os.path.exists(date_dir):
            print(f"Directory {date_dir} does not exist, skipping...")
            continue

        sequences = [d for d in os.listdir(date_dir) if os.path.isdir(os.path.join(date_dir, d)) and d.endswith('_sync')]

        for seq in sequences:
            seq_path = os.path.join(date_dir, seq)
            print(seq_path)
            output_file = os.path.join(output_dir, f"{os.path.basename(seq_path)}.bag")
            print(f"Converting {seq_path} to ROS bag...")

            try:
                subprocess.run([
                    'rosrun', 
                    'kitti_to_rosbag',
                    'kitti_rosbag_converter',
                    date_dir,
                    seq_path,
                    output_file
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error converting {seq_path}: {e}")
                continue

if __name__ == "__main__":
    KITTIDIR = '/home/adam/Downloads/raw_data_kitti'  # Update this path to your KITTI raw data directory
    OUTPUTDIR = '/media/adam/Data/ood_slam_data/datasets/kitti_bag_files'  # Update this path to your desired output directory
    DATES = ["2011_09_30", "2011_10_03"]  # Add more dates as needed

    if not os.path.exists(OUTPUTDIR):
        os.makedirs(OUTPUTDIR)

    convert_kitti_to_bag(KITTIDIR, OUTPUTDIR, DATES)
