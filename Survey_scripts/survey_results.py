import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def scale_to_01(arr):
    min = arr.min()
    max = arr.max()
    return (arr - min) / (max - min)

def plot_mse(mse, title):
    plt.figure(figsize=(10, 5))  # Set figure size for better visibility
    plt.scatter(range(len(mse)), mse, color='blue', marker='o')  # Scatter plot with specified color and marker
    plt.xlabel("Users")  # Corrected x-label
    plt.ylabel("MSE")  # Corrected y-label
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.xticks(range(len(mse)))  # Adjust x-ticks based on the number of users
    plt.savefig(title + ".png")
    plt.close()

def plot_comparison(rampa, kns, title,maximum):
    num_questions = np.shape(rampa)[1]
    positions = np.arange(num_questions)
    
    data = {
        'Question': np.tile(np.arange(1, num_questions + 1), 21 * 2),  # Questions 1-9 repeated
        'Score': np.concatenate((rampa, kns)).flatten(),
        'Method': ['RAMPA'] * rampa.size + ['Kinesthetic Teaching'] * kns.size
    }

    df = pd.DataFrame(data)

    # Create the boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Question', y='Score', hue='Method', data=df, 
                width=0.5,  # Adjust box width
                dodge=True,  # To separate the boxes for each method
                palette='Set2'  # Use a color palette
                )
    
    plt.title(title)
    plt.xlabel('Questions')
    plt.ylabel('Scores')

    plt.yticks(np.arange(0,maximum) )
    plt.xticks(positions, [f'Q{i+1}' for i in range(num_questions)])
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.grid()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(title + ".png")
    plt.close()  # Close the figure after saving


def find_mse(data,mean_data):
    result = []
    for i in range(len(data)):
        mse = np.square(scale_to_01(data[i])-mean_data).mean()
        result.append(mse)
    return np.array(result)


csv_file_path = '/home/ur-colors/Desktop/RAMPA/Survey_scripts/RAMPA - Questionnaire.csv'  
data = pd.read_csv(csv_file_path)

general_data = data.iloc[:, 10:28].to_numpy()
rampa_general = general_data[:, :10]
kns_general1 = general_data[:, 10:14]
kns_general2 = general_data[:, 14:]
kns_general3 = np.empty((21, 2))
kns_general = np.concatenate((kns_general1,kns_general3,kns_general2), axis=-1)
print(np.shape(kns_general))

ueq_data = data.iloc[:, 28:36].to_numpy()

tlx_data = data.iloc[:,36:48].to_numpy()
rampa_tlx = tlx_data[:, :6]
kns_tlx = tlx_data[:, 6:]

total_data = np.concatenate((general_data,tlx_data), axis=-1)
rampa_total = np.concatenate((rampa_general,rampa_tlx), axis=-1)
kns_total = np.concatenate((kns_general,kns_tlx), axis=-1)

mean_general = np.mean(scale_to_01(general_data),axis=0)
mean_tlx = np.mean(scale_to_01(tlx_data),axis=0)
mean_total = np.mean(scale_to_01(total_data),axis=0)

ueq = data.iloc[:, 28:36].to_numpy()
mean_ueq = np.mean(scale_to_01(ueq),axis = 0)


rampa_general_mean = np.mean(rampa_general, axis=0) - 1
sus_score = 2.5 * (20 + np.sum(rampa_general_mean * [1,-1,1,-1,1,-1,1,-1,1,-1]))
print(sus_score)

kns_general_mean = np.mean(np.concatenate((kns_general1, kns_general2), axis=-1), axis=0)
kns_sus_score = (100/32) * (16 + np.sum(kns_general_mean * [1,-1,1,-1,1,-1,1,-1]))
print(kns_sus_score)

"""
flexible_data = np.array([2,3,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5])
print(f" flexible mean: {np.mean(flexible_data)}, std: {np.std(flexible_data)}")

progress_data = np.array([3,3,3,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5])
print(f"progress mean: {np.mean(progress_data)}, std: {np.std(progress_data)}")
"""




mse_ueq = find_mse(ueq,mean_ueq)
mse_general = find_mse(general_data,mean_general)
mse_tlx = find_mse(tlx_data,mean_tlx)
mse_total = find_mse(total_data,mean_total)






#plot_mse(mse_general,"MSE of Users for General Questions")
#plot_mse(mse_tlx,"MSE of Users for NASA-TLX Questions")
#plot_mse(mse_total,"MSE of Users for all Questions")
#plot_comparison(rampa_general,kns_general,"Comparison of Methods for General Questions",5)
#plot_comparison(rampa_tlx,kns_tlx,"Comparison of Methods for NASA-TLX Questions",10)
"""
print(np.shape(ueq_data))

plt.figure(figsize=(8, 6))
sns.boxplot(ueq_data,width = 0.3,color = "orange",dodge= True)

sns.boxplot(x='Question', y='Score', hue='Method', data=df, 
            width=0.5,  # Adjust box width
            dodge=True,  # To separate the boxes for each method
            palette='Set2'  # Use a color palette
                )

title = "User Experience Questionnaire Results of RAMPA"  
maximum = 8
num_questions = 8
positions = np.arange(8) 
plt.title(title)
plt.xlabel('Questions')
plt.ylabel('Scores')

plt.yticks(np.arange(0,maximum) )
plt.xticks(positions, [f'Q{i+1}' for i in range(num_questions)])
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
plt.grid()
plt.tight_layout()

# Save the figure
plt.savefig(title + ".png")
plt.close()  # Close the figure after saving

"""


