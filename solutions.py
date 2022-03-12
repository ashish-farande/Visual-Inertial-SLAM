from utils import *
from utils import *

OBSERVATION_NOISE = 0.05


def imu_localization(timestamp, angular_velocity, linear_velocity):
    trajectory = np.zeros((4, 4, timestamp.shape[1]))
    pose_mean = build_pose(np.identity(3), np.zeros((3, 1)))
    pose_covariance = np.identity(6) * 0.05
    trajectory[:, :, 0] = pose_mean

    for t in range(1, timestamp.shape[1]):
        delta_t = timestamp[0, t] - timestamp[0, t - 1]
        angular_velocity_t = build_skew(angular_velocity[:, t])
        linear_velocity_t = linear_velocity[:, t][:, np.newaxis]

        pose_hat = build_twist(angular_velocity_t, linear_velocity_t)
        # pose_curve_hat = build_curve_hat(angular_velocity_t, build_skew(linear_velocity_t))

        pose_mean = pose_mean @ scipy.linalg.expm(delta_t * pose_hat)

        # Covariance is playing no role over here
        # pose_covariance = scipy.linalg.expm(-delta_t * pose_curve_hat) @ pose_covariance @ scipy.linalg.expm(-delta_t * pose_curve_hat)

        trajectory[:, :, t] = pose_mean

    return trajectory


def landmark_mapping(timestamp, k_s, cam_T_imu, landmarks, trajectory):
    landmark_count = landmarks.shape[1]
    landmark_poses = np.zeros((3 * landmark_count, 1))
    landmark_covariance = np.identity(3 * landmark_count)
    landmark_visited = np.zeros(landmark_count)

    for t in range(1, timestamp.shape[1]):
        if t % 100 == 0:
            print(t)

        imu_T_world = np.linalg.inv(trajectory[:, :, t])

        # Landmarks for the current timestamp
        landmarks_t = np.squeeze(landmarks[:, :, t])
        landmark_ind_t = np.where(landmarks_t[0, :] != -1)[0]

        non_visited_feat_index = []
        for ind in landmark_ind_t:
            if landmark_visited[ind] == 0:
                # Initialize the poses
                landmark_poses[3 * ind:3 * (ind + 1)] = convert_to_worldframe(k_s, cam_T_imu, imu_T_world, landmarks_t[:, ind])
                landmark_covariance[3 * ind:3 * (ind + 1), 3 * ind:3 * (ind + 1)] = np.identity(3) * 0.001
                landmark_visited[ind] = 1
            else:
                non_visited_feat_index.append(ind)
        val_feat_ind = np.array(non_visited_feat_index)

        N_t = val_feat_ind.shape[0]
        if N_t > 0:
            # Poses for the current observations
            poses_t = landmark_poses.reshape(landmark_count, 3).T[:, val_feat_ind]

            # Calculate the innovation
            observations = landmarks_t[:, val_feat_ind].T.flatten()
            prediction = predict_features(k_s, cam_T_imu, imu_T_world, poses_t).T.flatten()
            innovation = (observations - prediction)[:, np.newaxis]

            # Calculate the derivative matrix
            H_t = np.zeros((4 * N_t, 3 * landmark_count))
            for i in range(N_t):
                j = val_feat_ind[i]
                H_t[4 * i:4 * (i + 1), 3 * j:3 * (j + 1)] = get_h_matrix(k_s, cam_T_imu, imu_T_world, landmark_poses[3 * j:3 * (j + 1)])

            # updates
            kalman_gain = landmark_covariance @ H_t.T @ np.linalg.inv(H_t @ landmark_covariance @ H_t.T + np.identity(4 * N_t) * OBSERVATION_NOISE)
            landmark_poses = landmark_poses + kalman_gain @ innovation
            landmark_covariance = (np.identity(3 * landmark_count) - kalman_gain @ H_t) @ landmark_covariance

    landmark_poses = landmark_poses.reshape(landmark_count, 3).T
    new_landmarks = [landmark_poses[:, 0]]
    for i in range(1, landmark_poses.shape[1]):
        if np.linalg.norm(landmark_poses[:, i]) < 2000:
            new_landmarks.append(landmark_poses[:, i])

    return np.array(new_landmarks)


def visual_inertial_slam(timestamp, angular_velocity, linear_velocity, k_s, cam_T_imu, landmarks):

    pass