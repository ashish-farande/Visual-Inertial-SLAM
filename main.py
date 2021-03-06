import os

from slam import *

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# The Vehicle will know details about the sensors and its data
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
TRAJECTORY_1_PATH = os.path.join(DATA_PATH, '10.npz')
TRAJECTORY_2_PATH = os.path.join(DATA_PATH, '03.npz')

if __name__ == '__main__':
    # Load the measurements
    timestamp, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(TRAJECTORY_1_PATH)

    k_s = get_stereo_matrix(K, b)
    cam_T_imu = np.linalg.inv(imu_T_cam)
    reduced_features = features[:, range(0, features.shape[1], 50), :]

    # (a) IMU Localization via EKF Prediction
    trajectory = imu_localization(timestamp, angular_velocity, linear_velocity)

    # (b) Landmark Mapping via EKF Update
    landmarks = landmark_mapping(timestamp, k_s, cam_T_imu, reduced_features, trajectory)

    # (c) Visual-Inertial SLAM
    trajectory_v, landmarks_v = visual_inertial_slam(timestamp, angular_velocity, linear_velocity, k_s, cam_T_imu, reduced_features)

    # You can use the function below to visualize the robot pose over time
    visualize_trajectory_2d(trajectory, path_name="Path_10", show_ori=True, features=landmarks)
    visualize_trajectory_2d(trajectory_v, path_name="Path_10", show_ori=True, features=landmarks_v)

    compare_trajectory(trajectory, trajectory_v, path1_name="LAM", path2_name="SLAM", show_ori=False, save_name="feat")
    compare_trajectory(trajectory, trajectory_v, path1_name="LAM", path2_name="SLAM", show_ori=False, features1=landmarks, features2=landmarks_v, save_name="no_feat")
