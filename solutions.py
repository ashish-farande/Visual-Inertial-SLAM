import scipy.linalg

from utils import *

L_NOISE = 0.05
W_NOISE = 0.01
V_SCALE = 1
COV_INIT = 0.001

def imu_localization(timestamp, angular_velocity, linear_velocity):
    trajectory = np.zeros((4, 4, timestamp.shape[1]))
    pose_mean = build_pose(np.identity(3), np.zeros((3, 1)))
    pose_covariance = np.identity(6) * 0.05
    trajectory[:, :, 0] = pose_mean

    for t in range(1, timestamp.shape[1]):
        delta_t = timestamp[0, t] - timestamp[0, t - 1]
        angular_velocity_t = angular_velocity[:, t]
        linear_velocity_t = linear_velocity[:, t][:, np.newaxis]

        pose_hat = build_twist(angular_velocity_t, linear_velocity_t)
        # pose_curve_hat = build_curve_hat(angular_velocity_t, build_skew(linear_velocity_t))

        pose_mean = pose_mean @ scipy.linalg.expm(delta_t * pose_hat)

        # Covariance is playing no role over here
        # pose_covariance = scipy.linalg.expm(-delta_t * pose_curve_hat) @ pose_covariance @ scipy.linalg.expm(-delta_t * pose_curve_hat)

        trajectory[:, :, t] = pose_mean

    return trajectory


def landmark_mapping(timestamp, k_s, cam_T_imu, landmarks, trajectory, verbose=True):
    landmark_count = landmarks.shape[1]
    landmark_mean = np.zeros((3 * landmark_count, 1))
    landmark_covariance = np.identity(3 * landmark_count)
    landmark_visited = np.zeros(landmark_count)

    for t in range(1, timestamp.shape[1]):
        if verbose and t % 100 == 0:
            print(t)

        imu_T_world = np.linalg.inv(trajectory[:, :, t])

        # Landmarks for the current timestamp
        landmarks_t = np.squeeze(landmarks[:, :, t])
        landmark_ind_t = np.where(landmarks_t[0, :] != -1)[0]

        non_visited_feat_index = []
        for ind in landmark_ind_t:
            if landmark_visited[ind] == 0:
                # Initialize the poses
                landmark_mean[3 * ind:3 * (ind + 1)] = convert_to_worldframe(k_s, cam_T_imu, imu_T_world, landmarks_t[:, ind])
                landmark_covariance[3 * ind:3 * (ind + 1), 3 * ind:3 * (ind + 1)] = np.identity(3) * 0.001
                landmark_visited[ind] = 1
            else:
                non_visited_feat_index.append(ind)
        val_feat_ind = np.array(non_visited_feat_index)

        N_t = val_feat_ind.shape[0]
        if N_t > 0:
            # Poses for the current observations
            poses_t = landmark_mean.reshape(landmark_count, 3).T[:, val_feat_ind]

            # Calculate the innovation
            observations = landmarks_t[:, val_feat_ind].T.flatten()
            prediction = convert_to_pixel(k_s, cam_T_imu, imu_T_world, poses_t).T.flatten()
            innovation = (observations - prediction)[:, np.newaxis]

            # Calculate the derivative matrix
            H_t = np.zeros((4 * N_t, 3 * landmark_count))
            for i in range(N_t):
                j = val_feat_ind[i]
                H_t[4 * i:4 * (i + 1), 3 * j:3 * (j + 1)] = get_h_matrix(k_s, cam_T_imu, imu_T_world, landmark_mean[3 * j:3 * (j + 1)])

            # updates
            kalman_gain = landmark_covariance @ H_t.T @ np.linalg.inv(H_t @ landmark_covariance @ H_t.T + np.identity(4 * N_t) * V_SCALE)
            landmark_mean = landmark_mean + kalman_gain @ innovation
            landmark_covariance = (np.identity(3 * landmark_count) - kalman_gain @ H_t) @ landmark_covariance

    landmark_mean = landmark_mean.reshape(landmark_count, 3).T
    new_landmarks = [landmark_mean[:, 0]]
    for i in range(1, landmark_mean.shape[1]):
        if np.linalg.norm(landmark_mean[:, i]) < 2000:
            new_landmarks.append(landmark_mean[:, i])

    return np.array(new_landmarks).T


def visual_inertial_slam(timestamp, angular_velocity, linear_velocity, k_s, cam_T_imu, landmarks, verbose = True):
    # Motion Initialization
    trajectory = np.zeros((4, 4, timestamp.shape[1]))
    pose_mean = build_pose(np.identity(3), np.zeros((3, 1)))
    # trajectory[:, :, 0] = pose_mean

    # Observation Initialization
    landmark_count = landmarks.shape[1]
    landmark_mean = np.zeros((3 * landmark_count, 1))
    landmark_visited = np.zeros(landmark_count)

    # Initialize covariance
    covariance = np.identity(3 * landmark_count+6)
    # covariance[-6:, -6:] = np.identity(6)*0.01

    for t in range(1, timestamp.shape[1]):
        if verbose and t % 100 == 0:
            print(t)

        # Update Step
        imu_T_world = np.linalg.inv(pose_mean)

        # Landmarks for the current timestamp
        landmarks_t = np.squeeze(landmarks[:, :, t])
        landmark_ind_t = np.where(landmarks_t[0, :] != -1)[0]

        non_visited_feat_index = []
        for ind in landmark_ind_t:
            if landmark_visited[ind] == 0:
                # Initialize the poses
                landmark_mean[3 * ind:3 * (ind + 1)] = convert_to_worldframe(k_s, cam_T_imu, imu_T_world, landmarks_t[:, ind])
                covariance[3 * ind:3 * (ind + 1), 3 * ind:3 * (ind + 1)] = np.identity(3) * COV_INIT
                landmark_visited[ind] = 1
            else:
                non_visited_feat_index.append(ind)
        val_feat_ind = np.array(non_visited_feat_index)
        N_t = val_feat_ind.shape[0]


        if N_t > 0:
            # Poses for the current observations
            poses_t = landmark_mean.reshape(landmark_count, 3).T[:, val_feat_ind]

            # Calculate the innovation
            observations = landmarks_t[:, val_feat_ind].T.flatten()
            prediction = convert_to_pixel(k_s, cam_T_imu, imu_T_world, poses_t).T.flatten()
            innovation = (observations - prediction)[:, np.newaxis]

            # Calculate the derivative matrix
            H_t = np.zeros((4 * N_t, 3 * landmark_count + 6))
            for i in range(N_t):
                j = val_feat_ind[i]
                H_t[4 * i:4 * (i + 1), 3 * j:3 * (j + 1)] = get_h_matrix(k_s, cam_T_imu, imu_T_world, landmark_mean[3 * j:3 * (j + 1)])
                H_t[4 * i:4 * (i + 1), -6:] = get_pose_h_matrix(k_s, cam_T_imu, imu_T_world, landmark_mean[3 * j:3 * (j + 1)])

            # updates
            # kalman_gain = covariance @ H_t.T @ np.linalg.inv((H_t @ covariance @ H_t.T) + np.identity(4 * N_t) * V_SCALE)
            # covariance = (np.identity(3 * landmark_count+6) - (kalman_gain @ H_t)) @ covariance
            # kalman_gain_imu = kalman_gain[-6:,:]
            # kalman_gain_obs = kalman_gain[:-6,:]

            kalman_gain_obs = covariance[:-6,:-6] @ H_t[:,:-6].T @ np.linalg.inv(H_t[:,:-6] @ covariance[:-6,:-6] @ H_t[:,:-6].T + np.identity(4 * N_t) * V_SCALE)
            covariance[:-6,:-6] = (np.identity(3 * landmark_count) - (kalman_gain_obs @ H_t[:,:-6])) @ covariance[:-6,:-6]
            
            kalman_gain_imu = covariance[-6:,-6:] @ H_t[:,-6:].T @ np.linalg.inv(H_t[:,-6:] @ covariance[-6:,-6:] @ H_t[:,-6:].T + np.identity(4 * N_t) * V_SCALE)
            covariance[-6:,-6:] = (np.eye(6) - (kalman_gain_imu @ H_t[:,-6:])) @ covariance[-6:,-6:]

            # mean updates
            landmark_mean = landmark_mean + kalman_gain_obs @ innovation

            d_pose = kalman_gain_imu @ innovation
            position = d_pose[:3]
            theta = d_pose[-3:]
            pose_mean = pose_mean @ scipy.linalg.expm(build_twist(theta, position))

        trajectory[:, :, t] = pose_mean

        # Prediction Step
        delta_t = timestamp[0, t] - timestamp[0, t - 1]
        angular_velocity_t = angular_velocity[:, t][:, np.newaxis] + add_noise(3, np.array([W_NOISE]))
        linear_velocity_t = linear_velocity[:, t][:, np.newaxis] + add_noise(3, np.array([L_NOISE]))

        pose_hat = build_twist(angular_velocity_t, linear_velocity_t)
        pose_mean = pose_mean @ scipy.linalg.expm(delta_t * pose_hat)

        pose_curve_hat = build_curly_hat(angular_velocity_t, linear_velocity_t)
        exp_cur = scipy.linalg.expm(-delta_t * pose_curve_hat)
        covariance[-6:, -6:] = exp_cur @ covariance[-6:, -6:] @ exp_cur.T

        # Store the trajectory

    landmark_mean = landmark_mean.reshape(landmark_count, 3).T
    new_landmarks = []
    for i in range(0, landmark_mean.shape[1]):
        if np.linalg.norm(landmark_mean[:, i]) < 2000:
            new_landmarks.append(landmark_mean[:, i])
    new_landmarks_tmp = np.array(new_landmarks)
    return trajectory, np.array(new_landmarks).T


