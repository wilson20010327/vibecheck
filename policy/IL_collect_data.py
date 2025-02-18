from ENV.env import RLEnv
from model.DQN import DQNAgent
import os
import numpy as np
import pandas as pd
import json


RANDOM = True
PERFECT=False
result_path = './result/IL/training/new-confusion-outdistribution-limit-init-0216/'
if os.path.exists(result_path):
    raise FileExistsError(f"The directory '{result_path}' already exists.")

os.makedirs(result_path)
print(f"Directory '{result_path}' created successfully.")
    
def get_true_action_from_env(env):
    euler=env.euler
    chat_to_index = {'z':0,'x':1,'f':2}
    return chat_to_index[env.get_gt_action(euler)]
   
def save_trajectory(trajectory, episode_num, path):
    """Save trajectory to a JSON file."""
    file_path = os.path.join(path, f"trajectory_episode_{episode_num}.csv")
    df = pd.DataFrame(trajectory)  # Convert trajectory to a DataFrame
    df.to_csv(file_path, index=False)  # Save to Excel
    print(f"Trajectory for episode {episode_num} saved to {file_path}")
    
def main():
    env=RLEnv(random_flag=RANDOM,perfect=PERFECT)
    sucess=0
    step_schedule = []
    big_triangular = 0
    # if not TEST:
    #     big_triangular=200
    epoch=1000
    for i in range (big_triangular):
        step_schedule.append(1000)
    for i in range (epoch-big_triangular):
        step_schedule.append(50)
        
    for j in range(epoch):
        trajectory = [] 
        tot_reward = 0
        env.reset()
        
        for i in range(step_schedule[j]):
            current = env.get_observation()
            real_data=env.euler
            action = get_true_action_from_env(env)
            next, reward, done, info = env.step(action)
            tot_reward+=reward
            # print('ground true:',env.current_state)
            # print('next:',next)
            # print('reward:',reward)
            # print('tot_reward:',tot_reward)
            # print('done:',done)
            trajectory.append({
                'step': i,
                'real_state': real_data,
                'state': current.tolist() if isinstance(current, np.ndarray) else current,
                'action': int(action),
                'reward': reward,
                'tot_reward': tot_reward,
                'next_state': next.tolist() if isinstance(next, np.ndarray) else next,
                'done': done,
            })
            if done:
                sucess+=1
                print('sucess:',sucess)
                break
        
        save_trajectory(trajectory, j, result_path)

    print('sucess:',sucess)
if __name__ == "__main__":
    main()    