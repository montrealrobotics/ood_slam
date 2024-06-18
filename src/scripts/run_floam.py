import os
import subprocess

def run_floam(input_bag, output_bag):
    launch_file = "/home/adam/catkin_ws/src/floam/launch/floam_mapping.launch"  # Update with the actual path
    command = [
        "roslaunch", launch_file,
        "input_bag:=" + input_bag,
        "output_bag:=" + output_bag
    ]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running FLOAM on {input_bag}: {e}")

if __name__ == "__main__":
    BAG_DIR = "/media/adam/Data/ood_slam_data/datasets/kitti_bag_files"  # Update with actual path
    OUTPUT_DIR = "/media/adam/Data/ood_slam_data/datasets/floam_output_bag_files"  # Update with actual path
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    bag_files = [os.path.join(BAG_DIR, f) for f in os.listdir(BAG_DIR) if f.endswith('.bag')]
    
    for bag_file in bag_files:
        output_bag = os.path.join(OUTPUT_DIR, os.path.basename(bag_file).replace('.bag', '_floam.bag'))
        run_floam(bag_file, output_bag)
