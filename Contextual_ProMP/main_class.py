import numpy as np
import os
import math
import matplotlib.pyplot as plt
from movement_primitives.promp import ProMP
from gmr import GMM


class T:

    def __init__(self):
        self.filenames = []
        self._demo_data = []
        self.isTraining = False
        self.last_training = ""
        self.sample_length = 100
        self.n_dims_pos = 3
        self.n_dims_or = 4
        self.gmm_context_pos = []
        self.gmm_context_or = []
        self.num_demo = 3

        self.priors = 20

        self.p_pos = ProMP(n_dims=self.n_dims_pos, n_weights_per_dim=20)
        self.p_or = ProMP(n_dims=self.n_dims_or, n_weights_per_dim=20)

        self.g_pos_x = GMM(n_components=self.priors, random_state=1234)
        self.g_pos_y = GMM(n_components=self.priors, random_state=1234)
        self.g_pos_z = GMM(n_components=self.priors, random_state=1234)
        self.g_or_x = GMM(n_components=self.priors, random_state=1234)
        self.g_or_y = GMM(n_components=self.priors, random_state=1234)
        self.g_or_z = GMM(n_components=self.priors, random_state=1234)
        self.g_or_w = GMM(n_components=self.priors, random_state=1234)


    def slerp(self, q1, q2, t):
        dot = np.dot(q1, q2)

        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)

        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)

        if sin_theta_0 > 0.001:
            theta = theta_0 * t
            sin_theta = np.sin(theta)

            s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s2 = sin_theta / sin_theta_0
        else:
            s1 = 1.0 - t
            s2 = t

        q_interp = s1 * q1 + s2 * q2

        return q_interp / np.linalg.norm(q_interp)

    def make_quaternions_continuous(self, trajectory):
        for i in range(1, len(trajectory)):
            prev_q = trajectory[i - 1][3:7]
            curr_q = trajectory[i][3:7]

            dot_product = np.dot(prev_q, curr_q)

            if dot_product < 0:
                trajectory[i][3:7] = -1 * curr_q

        return trajectory


    def interpolate_points(self, points, total_points):
        num_segments = len(points) - 1
        points_per_segment = total_points // num_segments
        remaining_points = total_points % num_segments

        interpolated_points = []
        points = np.array(points)
        for i in range(len(points) - 1):
            start_point = points[i]
            end_point = points[i + 1]
            num_points = points_per_segment + (1 if i < remaining_points else 0)

            for j in range(num_points):
                ratio = (j + 1) / (num_points + 1)  # Calculate ratio between start and end points
                interpolated_point = (1 - ratio) * start_point + ratio * end_point  # Linear interpolation formula
                interpolated_points.append(interpolated_point)
        return np.array(interpolated_points)


    def interpolate_quaternions(self,quaternions, total_points):
        num_segments = len(quaternions) - 1
        points_per_segment = (total_points - len(quaternions)) // num_segments
        remaining_points = (total_points - len(quaternions)) % num_segments

        interpolated_points = []
        for i in range(num_segments):
            q1 = quaternions[i]
            q2 = quaternions[i + 1]
            num_points = points_per_segment + (1 if i < remaining_points else 0)

            interpolated_points.append(q1)

            for j in range(num_points):
                t = (j + 1) / (num_points + 1)
                q = self.slerp(q1, q2, t)
                interpolated_points.append(q)

        interpolated_points.append(quaternions[-1])

        interpolated_points = np.array(interpolated_points)

        return interpolated_points


    def start_training(self, input, trajectories, context = []):
        global last_training
        global _demo_data

        number_of_demonstrations = len(trajectories)
        all_demonstrations = []
        for trajectory in trajectories:
            pos_trajectory = [pose[0:3] for pose in trajectory]
            or_trajectory = [pose[3:7] for pose in trajectory]
            interpolated_points = self.interpolate_points(pos_trajectory, self.sample_length)
            interpolated_quaternions = self.interpolate_quaternions(or_trajectory, self.sample_length)

            interpolated_data = np.concatenate((interpolated_points, interpolated_quaternions), axis=-1)

            all_demonstrations.append(interpolated_data)
        demo_data = np.array(all_demonstrations)
        demo_data = demo_data.reshape((number_of_demonstrations, self.sample_length, self.n_dims_pos + self.n_dims_or))

        demo_data_pos = demo_data[:, :, :3]
        demo_data_or = demo_data[:, :, -4:]

        _demo_data = demo_data

        global num_demo
        num_demo = number_of_demonstrations

        if input == "promp":
            Ts = np.linspace(0, 1, self.sample_length).reshape((1, self.sample_length))  # Generate Ts for one demonstration
            Ts = np.tile(Ts, (number_of_demonstrations, 1))
            # Ts = np.linspace(0,1,sample_length).reshape((number_of_demonstrations,sample_length))
            Ypos = demo_data_pos
            Yor = demo_data_or

            # Training
            self.p_pos.imitate(Ts, Ypos)
            self.p_or.imitate(Ts, Yor)

            last_training = "promp"

        elif (input == "gmm"):

            Ypos_x = demo_data_pos[:, :, 0]
            Ypos_y = demo_data_pos[:, :, 1]
            Ypos_z = demo_data_pos[:, :, 2]

            Yor_x = demo_data_or[:, :, 0]
            Yor_y = demo_data_or[:, :, 1]
            Yor_z = demo_data_or[:, :, 2]
            Yor_w = demo_data_or[:, :, 3]

            # Training
            self.g_pos_x.from_samples(Ypos_x)
            self.g_pos_y.from_samples(Ypos_y)
            self.g_pos_z.from_samples(Ypos_z)
            self.g_or_x.from_samples(Yor_x)
            self.g_or_y.from_samples(Yor_y)
            self.g_or_z.from_samples(Yor_z)
            self.g_or_w.from_samples(Yor_w)

            last_training = "gmm"

        elif (input == "contextual_promp"):
            timesteps = np.linspace(0, 1, self.sample_length)
            timesteps = np.tile(timesteps, (number_of_demonstrations, 1))
            weights_pos = np.empty((number_of_demonstrations, self.n_dims_pos * self.priors))
            weights_or = np.empty((number_of_demonstrations, self.n_dims_or * self.priors))
    
            self.p_pos.imitate(timesteps, demo_data_pos)
            self.p_or.imitate(timesteps, demo_data_or)
    
            for demo_idx in range(number_of_demonstrations):
                weights_pos[demo_idx] = self.p_pos.weights(timesteps[demo_idx], demo_data_pos[demo_idx]).flatten()
                weights_or[demo_idx] = self.p_or.weights(timesteps[demo_idx], demo_data_or[demo_idx]).flatten()
    
            # weights = np.concatenate((weights_pos, weights_or), axis=-1)
    
            context_features = context.reshape(-1, 1)  # Reshape to column vector so it can be concatenated
            X_pos = np.hstack((context_features, weights_pos))
            X_or = np.hstack((context_features, weights_or))
    
            global gmm_context_pos
            global gmm_context_or
            gmm_context_pos = GMM(n_components=3, random_state=0)
            gmm_context_or = GMM(n_components=3, random_state=0)
            gmm_context_pos.from_samples(X_pos)
            gmm_context_or.from_samples(X_or)
    
            last_training = "contextual_promp"

        isTraining = False

    def condition_model(self,waypoint,novel_context = []):
        global last_training
        condition_pose = waypoint[0:3]
        condition_orientation = waypoint[3:7]
        sample_length = self.sample_length

        trajectory_pos = []
        trajectory_or = []

        if (last_training == "promp"):
            # conditioning model
            self.p_pos.condition_position([condition_pose[0], condition_pose[1], condition_pose[2]], t=0.5)

            # sampling
            # trajectory_pos = p_pos.sample_trajectories(T=np.linspace(0, 1, sample_length).reshape(sample_length),
            #                                         n_samples=1, random_state=np.random.RandomState(seed=1234))

            self.p_or.condition_position([condition_orientation[0], condition_orientation[1], condition_orientation[2], condition_orientation[3]], t=0.5)

            # sampling
            # trajectory_or = p_or.sample_trajectories(T=np.linspace(0, 1, sample_length).reshape(sample_length), n_samples=1,
            #                                        random_state=np.random.RandomState(seed=1234))
        
        elif (last_training == "gmm"):

            self.g_pos_x.condition([sample_length/2], condition_pose[0])
            self.g_pos_y.condition([sample_length/2], condition_pose[1])
            self.g_pos_z.condition([sample_length/2], condition_pose[2])

            self.g_or_x.condition([sample_length/2], condition_orientation[0])
            self.g_or_y.condition([sample_length/2], condition_orientation[1])
            self.g_or_z.condition([sample_length/2], condition_orientation[2])
            self.g_or_w.condition([sample_length/2], condition_orientation[3])

        elif (last_training == "contextual_promp"):

            timesteps = np.linspace(0, 1, sample_length)
            timesteps = np.tile(timesteps, (num_demo, 1))

            new_context_feature = novel_context  # Novel context
            pmp_pos = self.get_promp_from_context(gmm_context_pos, new_context_feature, True)
            pmp_or = self.get_promp_from_context(gmm_context_or, new_context_feature, False)

            return pmp_pos,pmp_or

        
    def get_promp_from_context(self,gmm_, context_feature, isPos):
        n_dims = 0
        priors = self.priors
        if (isPos):
            n_dims = 3
        else:
            n_dims = 4
        pmp = ProMP(n_dims=n_dims, n_weights_per_dim=priors)
        context_feature = np.array([context_feature]).reshape(1, -1)
        conditional_weight_distribution = gmm_.condition(np.arange(context_feature.shape[1]), context_feature).to_mvn()
        conditional_mean = conditional_weight_distribution.mean[-(n_dims) * priors:]
        conditional_covariance = conditional_weight_distribution.covariance[-(n_dims) * priors:, -(n_dims) * priors:]
        pmp.from_weight_distribution(conditional_mean, conditional_covariance)
        return pmp