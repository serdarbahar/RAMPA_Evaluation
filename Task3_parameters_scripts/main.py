import numpy as np
import os 
import glob
import matplotlib.pyplot as plt
import utils 


#FIX ACCORDINGLY
base_path = "/home/ur-colors/Desktop/Rampa_Trajectories"

user_input = input("Enter the name of the user or 'all': ")

if user_input == "all":
    hand_parameters = []
    ar_parameters = []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path,folder_name)
        hand_trajectories, ar_trajectories = utils.load_trajectories(folder_path)

    
        for traj in hand_trajectories:
            j,d,v = utils.compute_parameters(traj)
        hand_parameters.append([j,d,v])
    

        for traj in ar_trajectories:
            j,d,v = utils.compute_parameters(traj)
        ar_parameters.append([j,d,v])
    utils.plot_results(np.array(hand_parameters),np.array(ar_parameters))
else:
    utils.find_result_one_user(os.path.join(base_path,user_input), plot = True)

