import os
import glob
import numpy as np
from main_class import T

def scale_to_01(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def change_hand_trajectory(x):
    x[3], x[4], x[5], x[6] = x[4], x[6], x[3], x[5]
    return x

def chang_AR_trajectory(traj):
    traj = traj[:,:7]
    return traj


def compute_mse(data1,data2,normalize = False):
    if normalize:
        mse = np.square(scale_to_01(data1)-scale_to_01(data2)).mean()
    else:
        mse = np.square(data1-data2).mean()
    return mse

def load_trajectories(folder_path):

    if os.path.isdir(folder_path):
        hand_trajectories = []
        ar_trajectories = []

        hand_files = glob.glob(os.path.join(folder_path,"hand_trajectory*"))
        ar_files = glob.glob(os.path.join(folder_path,"ar_trajectory*"))

        for file in hand_files:
            traj = change_hand_trajectory(np.load(file))
            hand_trajectories.append(traj)
            

        for file in ar_files:
            traj = chang_AR_trajectory(np.load(file))
            ar_trajectories.append(traj)
            
        
        return hand_trajectories,ar_trajectories
    else:
        print("Folder not found")

def load_waypoints(folder_path):
    if os.path.isdir(folder_path):
        hand_waypoint = np.load(os.path.join(folder_path,"RealWaypoint.npy"))
        AR_waypoint = np.load(os.path.join(folder_path,"AR_waypoint.npy"),allow_pickle=True)
        
        ar_waypoint = np.array([AR_waypoint[0].x, AR_waypoint[0].y, AR_waypoint[0].z, AR_waypoint[1].x, AR_waypoint[1].y, AR_waypoint[1].z, AR_waypoint[1].w])
        return hand_waypoint.flatten(),ar_waypoint
    else:
        print("Folder not found")


def find_trajectories(traj,wp):
    model = T()
    timesteps = np.linspace(0, 1, model.sample_length)
    timesteps = np.tile(timesteps, (model.num_demo, 1))
    
    model.start_training("promp", traj)
    model.condition_model(wp)

    p_pos = model.p_pos
    p_or = model.p_or

    traj_conditioned_by_user_pos = p_pos.mean_trajectory(timesteps[0])
    traj_conditioned_by_user_or = p_or.mean_trajectory(timesteps[0])

    context = np.array([0.145, 0.24, 0.31])
    model = T()
    model.start_training("contextual_promp",traj,context)
    novel_context = np.array([0.193])
    cp_pos, cp_or = model.condition_model(wp,novel_context)

    traj_conditioned_by_context_pos = cp_pos.mean_trajectory(timesteps[0])
    traj_conditioned_by_context_or = cp_or.mean_trajectory(timesteps[0])

    traj_conditioned_by_user = np.concatenate((traj_conditioned_by_user_pos,traj_conditioned_by_user_or),axis=-1)
    traj_conditioned_by_context = np.concatenate((traj_conditioned_by_context_pos,traj_conditioned_by_context_or),axis=-1)

    return traj_conditioned_by_user_pos, traj_conditioned_by_context_pos

def find_mse(traj,wp,normalize=False):
    #model = T()
    traj1,traj2 = find_trajectories(traj,wp)
    mse = compute_mse(traj1,traj2,normalize)
    return mse