import numpy as np
import os 
import glob
import matplotlib.pyplot as plt

def change_hand_trajectory(x):
    x[3], x[4], x[5], x[6] = x[4], x[6], x[3], x[5]
    return x

def chang_AR_trajectory(traj):
    traj = traj[:,:7]
    return traj

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
    
def compute_parameters(traj):

    jerk = np.diff(traj,3,axis=0)
    jerk_mean = np.mean(np.linalg.norm(jerk,axis=1))

    mean_traj = np.mean(traj,axis=0)
    avg_deviation = np.mean(np.abs(traj - mean_traj))

    variation = np.mean(np.var(traj,axis=0))
    
    return jerk_mean, avg_deviation, variation

def find_parameters(hand_trajectories,ar_trajectories):
    
    hand_parameters = []
    ar_parameters = []
    for traj in hand_trajectories:
        j,d,v = compute_parameters(traj)
        hand_parameters.append([j,d,v])
    

    for traj in ar_trajectories:
        j,d,v = compute_parameters(traj)
        ar_parameters.append([j,d,v])

    return np.array(hand_parameters), np.array(ar_parameters)

def plot_results(hand_parameters,ar_parameters):

    plt.figure(figsize=(14,6))
    plt.style.use("seaborn-darkgrid")

    plt.subplot(1,3,1)
    plt.boxplot([hand_parameters[:,0],ar_parameters[:,0]], labels = ["Hand Trajectories", "AR trajectories"])
    plt.title("Jerk Comparision")
    plt.ylabel("Jerk")

    plt.subplot(1,3,2)
    plt.boxplot([hand_parameters[:,1],ar_parameters[:,1]], labels = ["Hand Trajectories", "AR trajectories"])
    plt.title("Average Deviation Comparision")
    plt.ylabel("Average Deviation")

    plt.subplot(1,3,3)
    plt.boxplot([hand_parameters[:,2],ar_parameters[:,2]], labels = ["Hand Trajectories", "AR trajectories"])
    plt.title("Variation Comparision")
    plt.ylabel("Variation")

    plt.tight_layout()
    plt.show()

def find_result_one_user(folder_path, plot = False):

    hand_trajectories, ar_trajectories = load_trajectories(folder_path)
    hand_parameters, ar_parameters = find_parameters(hand_trajectories,ar_trajectories)
    if plot == True:
        plot_results(hand_parameters,ar_parameters)

    return hand_trajectories, ar_trajectories, hand_parameters, ar_parameters
    




