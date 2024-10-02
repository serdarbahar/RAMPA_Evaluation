import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_file_path = '/home/ur-colors/Desktop/RAMPA/TaskCompletionTime_scripts/times.csv'  
data = pd.read_csv(csv_file_path)

data = data.to_numpy()
raw_data = data[:,[0,2,4,6,8,9]] #without num of trials

task1_data_hand = raw_data[:,:1]
task1_data_ar = raw_data[:,[1]]
task2_data_hand = raw_data[:,[2]]
task2_data_ar = raw_data[:,[3]]
task3_data_hand = raw_data[:,[4]]
task3_data_ar = raw_data[:,[5]]

data_task1 = {
    'Value': np.concatenate([task1_data_hand.flatten(), task1_data_ar.flatten()]),
    'Source': ['Kinesthetic Control'] * 20 + ['RAMPA'] * 20,
    'Task': 'Task 1'
}

data_task2 = {
    'Value': np.concatenate([task2_data_hand.flatten(), task2_data_ar.flatten()]),
    'Source': ['Kinesthetic Control'] * 20 + ['RAMPA'] * 20,
    'Task': 'Task 2'
}

data_task3 = {
    'Value': np.concatenate([task3_data_hand.flatten(), task3_data_ar.flatten()]),
    'Source': ['Kinesthetic Control'] * 20 + ['RAMPA'] * 20,
    'Task': 'Task 3'
}

df_task1 = pd.DataFrame(data_task1)
df_task2 = pd.DataFrame(data_task2)
df_task3 = pd.DataFrame(data_task3)

fig, axes = plt.subplots(1, 3, figsize=(9, 6))

# Plot for Task 1
sns.boxplot(x='Source', y='Value', 
            data=df_task1, ax=axes[0],width=0.6,
            flierprops={"marker": "o", 'markerfacecolor': 'red', 'markeredgecolor': 'r'})
axes[0].set_ylabel("Time (s)")
axes[0].set_xlabel("")
axes[0].set_title('Task 1 Completion Time')
axes[0].grid(True)  # Add grid

# Plot for Task 2
sns.boxplot(x='Source', y='Value', 
            data=df_task2, ax=axes[1],width=0.6,
            flierprops={"marker": "o", 'markerfacecolor': 'red', 'markeredgecolor': 'r'})
axes[1].set_ylabel("Time (s)")
axes[1].set_xlabel("")
axes[1].set_title('Task 2 Completion Time')
axes[1].grid(True)  # Add grid

# Plot for Task 3
sns.boxplot(x='Source', y='Value', 
            data=df_task3, ax=axes[2],width=0.6,
            flierprops={"marker": "o", 'markerfacecolor': 'red', 'markeredgecolor': 'r'})
axes[2].set_ylabel("Time (s)")
axes[2].set_xlabel("")
axes[2].set_title('Task 3 Completion Time')
axes[2].grid(True)  # Add grid

plt.tight_layout()

plt.savefig("task_completion_time.png")
plt.close()


task1_hand_expert, task1_hand_novice = task1_data_hand[:10], task1_data_hand[10:]
task1_ar_expert, task1_ar_novice = task1_data_ar[:10], task1_data_ar[10:]

task2_hand_expert, task2_hand_novice = task2_data_hand[:10], task2_data_hand[10:]
task2_ar_expert, task2_ar_novice = task2_data_ar[:10], task2_data_ar[10:]

task3_hand_expert, task3_hand_novice = task3_data_hand[:10], task3_data_hand[10:]
task3_ar_expert, task3_ar_novice = task3_data_ar[:10], task3_data_ar[10:]

# Calculate mean and standard deviation for each subset
task1_hand_expert_mean, task1_hand_expert_std = np.mean(task1_hand_expert), np.std(task1_hand_expert)
task1_hand_novice_mean, task1_hand_novice_std = np.mean(task1_hand_novice), np.std(task1_hand_novice)

task1_ar_expert_mean, task1_ar_expert_std = np.mean(task1_ar_expert), np.std(task1_ar_expert)
task1_ar_novice_mean, task1_ar_novice_std = np.mean(task1_ar_novice), np.std(task1_ar_novice)

task2_hand_expert_mean, task2_hand_expert_std = np.mean(task2_hand_expert), np.std(task2_hand_expert)
task2_hand_novice_mean, task2_hand_novice_std = np.mean(task2_hand_novice), np.std(task2_hand_novice)

task2_ar_expert_mean, task2_ar_expert_std = np.mean(task2_ar_expert), np.std(task2_ar_expert)
task2_ar_novice_mean, task2_ar_novice_std = np.mean(task2_ar_novice), np.std(task2_ar_novice)

task3_hand_expert_mean, task3_hand_expert_std = np.mean(task3_hand_expert), np.std(task3_hand_expert)
task3_hand_novice_mean, task3_hand_novice_std = np.mean(task3_hand_novice), np.std(task3_hand_novice)

task3_ar_expert_mean, task3_ar_expert_std = np.mean(task3_ar_expert), np.std(task3_ar_expert)
task3_ar_novice_mean, task3_ar_novice_std = np.mean(task3_ar_novice), np.std(task3_ar_novice)

# Collecting results for expert and novice groups
results_expert_novice = {
    "Task 1 Hand Expert": {"Mean": task1_hand_expert_mean, "Std": task1_hand_expert_std},
    "Task 1 Hand Novice": {"Mean": task1_hand_novice_mean, "Std": task1_hand_novice_std},
    "Task 1 AR Expert": {"Mean": task1_ar_expert_mean, "Std": task1_ar_expert_std},
    "Task 1 AR Novice": {"Mean": task1_ar_novice_mean, "Std": task1_ar_novice_std},
    "Task 2 Hand Expert": {"Mean": task2_hand_expert_mean, "Std": task2_hand_expert_std},
    "Task 2 Hand Novice": {"Mean": task2_hand_novice_mean, "Std": task2_hand_novice_std},
    "Task 2 AR Expert": {"Mean": task2_ar_expert_mean, "Std": task2_ar_expert_std},
    "Task 2 AR Novice": {"Mean": task2_ar_novice_mean, "Std": task2_ar_novice_std},
    "Task 3 Hand Expert": {"Mean": task3_hand_expert_mean, "Std": task3_hand_expert_std},
    "Task 3 Hand Novice": {"Mean": task3_hand_novice_mean, "Std": task3_hand_novice_std},
    "Task 3 AR Expert": {"Mean": task3_ar_expert_mean, "Std": task3_ar_expert_std},
    "Task 3 AR Novice": {"Mean": task3_ar_novice_mean, "Std": task3_ar_novice_std},
}

print(results_expert_novice)