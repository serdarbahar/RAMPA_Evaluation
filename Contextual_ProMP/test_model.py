from main_class import T
import utils
import numpy as np
from movement_primitives.promp import ProMP
import os
import matplotlib.pyplot as plt
import time
import seaborn as sns


# FILL ACCORDINGLY
base_path = "/home/ur-colors/Desktop/Rampa_Trajectories"

hand_mse_values = []
ar_mse_values = []
normalize = True
num_user = 21
positions = np.arange(num_user)


for folder_name in os.listdir(base_path):
        
        folder_path = os.path.join(base_path,folder_name)
        hand_traj, ar_traj = utils.load_trajectories(folder_path)
        hand_wp,ar_wp = utils.load_waypoints(folder_path)

        hand_mse = utils.find_mse(hand_traj,hand_wp,normalize)  
        hand_mse_values.append(hand_mse)

        ar_mse = utils.find_mse(ar_traj,ar_wp,normalize)
        ar_mse_values.append(ar_mse)

        print(f"MSE of {folder_name} is computed.")
        
mean1 = np.mean(np.array(hand_mse_values))
mean2 = np.mean(np.array(ar_mse_values))
print(f"Mean of Hand MSEs: {mean1} Mean of AR MSEs : {mean2}")

plt.figure(figsize=(18, 6))
plt.scatter(range(21), hand_mse_values)
plt.scatter(range(21), ar_mse_values)
plt.legend(["Hand", "AR"])
plt.xticks(positions, [f'User{i+1}' for i in range(num_user)])
plt.ylim([0, 0.01])
plt.ylabel("MSE")
plt.grid()
plt.tight_layout()
plt.savefig("/home/ur-colors/Desktop/RAMPA/Contextual_ProMP/Comparison_of_MSE_values.png")  
plt.show()
plt.close()

data = [hand_mse_values, ar_mse_values]
plt.figure(figsize=(4, 8))
sns.boxplot(data=data,width= 0.5,palette="Set2",whis=(0, 100))
plt.xticks(np.arange(2),["Kinesthetic Teaching","RAMPA"])
plt.yticks(np.arange(6)/1000)
plt.title('Comparison of Two Methods')
plt.ylabel('MSE')
plt.grid()
plt.tight_layout()
plt.show()
plt.close() 














