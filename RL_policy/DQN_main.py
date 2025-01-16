from ENV.env import RLEnv
from model.DQN import DQNAgent
import os
import numpy as np
import pandas as pd
import json

TEST = True
result_path = './result/Jan_16_4/'
if not TEST and os.path.exists(result_path):
    raise FileExistsError(f"The directory '{result_path}' already exists.")
elif not TEST:
    os.makedirs(result_path)
    os.makedirs(result_path+"train/")
    os.makedirs(result_path+"test/")
    print(f"Directory '{result_path}' created successfully.")
    
def save_trajectory(trajectory, episode_num, path):
    """Save trajectory to a JSON file."""
    file_path = os.path.join(path, f"trajectory_episode_{episode_num}.csv")
    df = pd.DataFrame(trajectory)  # Convert trajectory to a DataFrame
    df.to_csv(file_path, index=False)  # Save to Excel
    print(f"Trajectory for episode {episode_num} saved to {file_path}")
    
def main():
    env=RLEnv()
    agent=DQNAgent(test=TEST,result_path=result_path,action_size=env.action_size,state_size=env.observation_size)
    sucess=0
    for j in range(1000):
        trajectory = [] 
        tot_reward = 0
        env.reset()
        
        for i in range(50):
            current = env.get_observation()
            action,greedy = agent.act(current)
            next, reward, done, info = env.step(action)
            loss='none'
            if not agent.test:
                loss=agent.step(current, action, reward, next, done)
            tot_reward+=reward
            # print('ground true:',env.current_state)
            # print('next:',next)
            # print('reward:',reward)
            # print('tot_reward:',tot_reward)
            # print('done:',done)
            trajectory.append({
                'step': i,
                'state': current.tolist() if isinstance(current, np.ndarray) else current,
                'action': int(action),
                'greedy': greedy,
                'reward': reward,
                'tot_reward': tot_reward,
                'next_state': next.tolist() if isinstance(next, np.ndarray) else next,
                'done': done,
                'loss': loss
            })
            if done:
                sucess+=1
                print('sucess:',sucess)
                break
        if TEST:
            save_trajectory(trajectory, j, result_path+"test/")
        if not TEST:
            save_trajectory(trajectory, j, result_path+"train/")

    if not agent.test:
        agent.save()
    print('sucess:',sucess)
if __name__ == "__main__":
    main()    