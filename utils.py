import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from transforms3d.euler import mat2euler


def load_data(file_name):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:
        t = data["time_stamps"]  # time_stamps
        features = data["features"]  # 4 x num_features : pixel coordinates of features
        linear_velocity = data["linear_velocity"]  # linear velocity measured in the body frame
        angular_velocity = data["angular_velocity"]  # angular velocity measured in the body frame
        K = data["K"]  # intrindic calibration matrix
        b = data["b"]  # baseline
        imu_T_cam = data["imu_T_cam"]  # Transformation from left camera to imu frame

    return t, features, linear_velocity, angular_velocity, K, b, imu_T_cam


def visualize_trajectory_2d(pose, path_name="Unknown", show_ori=False, features=None):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    n_pose = pose.shape[2]
    ax.plot(pose[0, 3, :], pose[1, 3, :], 'r-', label=path_name)
    ax.scatter(pose[0, 3, 0], pose[1, 3, 0], marker='s', label="start")
    ax.scatter(pose[0, 3, -1], pose[1, 3, -1], marker='o', label="end")

    if features is not None:
        ax.scatter(features[0, :], features[1, :], s=2, label="landmarks")

    if show_ori:
        select_ori_index = list(range(0, n_pose, max(int(n_pose / 50), 1)))
        yaw_list = []

        for i in select_ori_index:
            _, _, yaw = mat2euler(pose[:3, :3, i])
            yaw_list.append(yaw)

        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx, dy = [dx, dy] / np.sqrt(dx ** 2 + dy ** 2)
        ax.quiver(pose[0, 3, select_ori_index], pose[1, 3, select_ori_index], dx, dy, \
                  color="b", units="xy", width=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax


def add_noise(n, sigmas):
    n = np.random.normal(np.zeros(len(sigmas)), sigmas, size=(n, len(sigmas)))
    return n


def get_stereo_matrix(K, b):
    k_s = np.vstack((K[:2, :], K[:2, :]))
    k_s = np.hstack((k_s, np.zeros((4, 1))))
    k_s[2, -1] = - K[0, 0] * b
    return k_s


def build_skew(x):
    if isinstance(x, np.ndarray) and len(x.shape) >= 2:
        return np.array([[0, -x[2][0], x[1][0]],
                         [x[2][0], 0, -x[0][0]],
                         [-x[1][0], x[0][0], 0]])
    else:
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])


def build_twist(omega_hat, v):
    temp_pose = np.hstack((omega_hat, v))
    temp_pose = np.vstack((temp_pose, [0, 0, 0, 0]))
    return temp_pose


def build_pose(rotation, pose):
    temp_pose = np.hstack((rotation, pose))
    temp_pose = np.vstack((temp_pose, [0, 0, 0, 1]))
    return temp_pose


# def predict(in_delta_t, omega_t, v_t, old_pose, old_variance):
#     angular_velocity_t = build_skew(omega_t)
#     linear_velocity_t = v_t[:, np.newaxis]
#     pose_hat = build_pose_hat(angular_velocity_t, linear_velocity_t)
#     pose_curve_hat = build_curve_hat(angular_velocity_t, build_skew(linear_velocity_t))
#     new_mean = old_pose @ scipy.linalg.expm(in_delta_t * pose_hat)
#     new_variance = scipy.linalg.expm(-in_delta_t * pose_curve_hat) @ old_variance @ scipy.linalg.expm(-in_delta_t * pose_curve_hat)
#     return new_mean, new_variance


def build_curly_hat(omega_hat, v_hat):
    val_1 = np.hstack((omega_hat, v_hat))
    val_2 = np.hstack((np.zeros_like(omega_hat), omega_hat))
    val = np.vstack((val_1, val_2))
    return val


def predict_features(projection_matrix, cam_T_imu, imu_T_world, feature_poses):
    feature_poses = np.vstack((feature_poses, np.ones((1, feature_poses.shape[1]))))
    optical_frame = cam_T_imu @ imu_T_world @ feature_poses
    stereo_coord = projection_matrix @ (optical_frame / optical_frame[2, :])
    return stereo_coord


def get_h_matrix(projection_matrix, cam_T_imu, imu_T_world, feature_poses):
    P = np.vstack((np.identity(3), np.zeros((1, 3))))
    feature_poses = np.vstack((feature_poses, np.ones((1, feature_poses.shape[1]))))
    optical_frame = cam_T_imu @ imu_T_world @ feature_poses
    d_pi = d_canonical(optical_frame)
    h = projection_matrix @ d_pi @ cam_T_imu @ imu_T_world @ P
    return h


def d_canonical(optical_frame):
    d_pi = np.identity(4)
    d_pi[:, 2] = - np.squeeze(optical_frame / optical_frame[2, :])
    d_pi[2, 2] = 0
    d_pi /= optical_frame[2, 0]
    return d_pi


def convert_to_worldframe(k_s, cam_T_imu, imu_T_world, pixel):
    world_T_imu = np.linalg.inv(imu_T_world)
    imu_T_cam = np.linalg.inv(cam_T_imu)

    uL, vL, uR, vR = pixel
    fsu = k_s[0, 0]
    fsv = k_s[1, 1]
    cu = k_s[0, 2]
    cv = k_s[1, 2]
    fsub = -k_s[2, -1]

    d = uL - uR
    z = fsub / d
    x = z * (uL - cu) / fsu
    y = z * (vL - cv) / fsv

    m = (world_T_imu @ imu_T_cam @ np.array([x, y, z, 1]).reshape([4, 1]))[:3, :]
    return m
