
import rosbag
import numpy as np
from nav_msgs.msg import Odometry
import os

raw_to_sequence = {
    "2011_10_03_drive_0027_sync_floam.bag" : ("00.txt", 0, 4540),
    "2011_10_03_drive_0042_sync_floam.bag" : ("01.txt", 0, 1100),
    "2011_10_03_drive_0034_sync_floam.bag" : ("02.txt", 0, 4660),
    "2011_09_26_drive_0067_sync_floam.bag" : ("03.txt", 0, 800),
    "2011_09_30_drive_0016_sync_floam.bag" : ("04.txt", 0, 270),
    "2011_09_30_drive_0018_sync_floam.bag" : ("05.txt", 0, 2760),
    "2011_09_30_drive_0020_sync_floam.bag" : ("06.txt", 0, 1100),
    "2011_09_30_drive_0027_sync_floam.bag" : ("07.txt", 0, 1100),
    "2011_09_30_drive_0028_sync_floam.bag" : ("08.txt", 1100, 5170),
    "2011_09_30_drive_0033_sync_floam.bag" : ("09.txt", 0, 1590),
    "2011_09_30_drive_0034_sync_floam.bag" : ("10.txt", 0, 1200)
}

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    r11 = 1 - 2 * (qy**2 + qz**2)
    r12 = 2 * (qx*qy - qz*qw)
    r13 = 2 * (qx*qz + qy*qw)
    r21 = 2 * (qx*qy + qz*qw)
    r22 = 1 - 2 * (qx**2 + qz**2)
    r23 = 2 * (qy*qz - qx*qw)
    r31 = 2 * (qx*qz - qy*qw)
    r32 = 2 * (qy*qz + qx*qw)
    r33 = 1 - 2 * (qx**2 + qy**2)
    
    rotation_matrix = np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])
    return rotation_matrix

def parse_bag_file(bag_file_path, topic_name):
    bag = rosbag.Bag(bag_file_path)
    poses = []
    
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        tx = msg.pose.pose.position.x
        ty = msg.pose.pose.position.y
        tz = msg.pose.pose.position.z
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        poses.append((tx, ty, tz, qx, qy, qz, qw))
    
    bag.close()
    return poses

def convert_to_12_value(tx, ty, tz, qx, qy, qz, qw):
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    transformation_matrix = np.hstack((R, np.array([[tx], [ty], [tz]])))
    return transformation_matrix.flatten()

def transform_poses_to_origin(poses):
    # Transform all poses so that the first pose is at the origin
    origin_pose = poses[0]
    origin_translation = np.array(origin_pose[:3])
    origin_rotation = quaternion_to_rotation_matrix(*origin_pose[3:])

    transformed_poses = []

    for pose in poses:
        translation = np.array(pose[:3])
        rotation = quaternion_to_rotation_matrix(*pose[3:])

        relative_translation = translation - origin_translation
        relative_rotation = np.dot(origin_rotation.T, rotation)

        transformed_translation = np.dot(origin_rotation.T, relative_translation)
        transformed_pose = convert_to_12_value(
            *transformed_translation, 
            *rotation_to_quaternion(relative_rotation)
        )
        transformed_poses.append(transformed_pose)

    return transformed_poses

def rotation_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return qx, qy, qz, qw

def write_to_file(output_file_path, start, end, poses):
    i = 0
    with open(output_file_path, 'w') as file:
        for pose in poses:
            if i >= start and i <= end:
                line = ' '.join(map(lambda x: f"{x:.9e}", pose))
                file.write(line + '\n')
            i += 1

def main():
    bag_file_dir = '/media/adam/Data/ood_slam_data/datasets/floam_output_bag_files'
    output_dir = '/home/adam/Documents/research_internship/OOD_SLAM/src/results/floam_poses/data'
    topic_name = '/odom'  

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(bag_file_dir):
        if file_name.endswith('.bag'):
            bag_file_path = os.path.join(bag_file_dir, file_name)

            seq = raw_to_sequence[file_name]
            output_file_path = os.path.join(output_dir, seq[0])
            
            poses = parse_bag_file(bag_file_path, topic_name)
            #transformed_poses = transform_poses_to_origin(poses)
            write_to_file(output_file_path, seq[1], seq[2], poses)
            print(f"Converted poses written to {output_file_path}")

if __name__ == '__main__':
    main()
