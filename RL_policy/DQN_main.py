from ENV.env import RLEnv
from model.DQN import DQNAgent
import torch
import numpy as np
from collections import deque
def extract_deque(deque):
    temp=[]
    for i in range(len(deque)):
        temp.append(deque[i][0])
        temp.append(deque[i][1])
    return np.array(temp)
def main():
    env=RLEnv()
    agent=DQNAgent()
    tot_reward = 0
    state=deque([(-1,-1) for i in range(10)],maxlen=10)
    for i in range(100):
        state.append(env.current_state)
        current=extract_deque(state)
        print(current)
        action = agent.act(current)
        observation, reward, done, info = env.step(action)
        agent.step(current, action, reward, observation, done)
        tot_reward+=reward
        print('observation:',observation)
        print('reward:',reward)
        print('tot_reward:',tot_reward)
        print('done:',done)
        
        if done:
            break
if __name__ == "__main__":
    main()    