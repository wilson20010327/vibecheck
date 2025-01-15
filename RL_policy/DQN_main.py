from ENV.env import RLEnv
from model.DQN import DQNAgent
import torch
import numpy as np


def main():
    env=RLEnv()
    agent=DQNAgent()
    sucess=0
    for j in range(10):
        tot_reward = 0
        env.reset()
        for i in range(1000):
            current = env.get_observation()
            action = agent.act(current)
            print('action:',action)
            observation, reward, done, info = env.step(action)
            agent.step(current, action, reward, observation, done)
            tot_reward+=reward
            print('current_state:',env.current_state)
            print('observation:',observation)
            print('reward:',reward)
            print('tot_reward:',tot_reward)
            print('done:',done)
            
            if done:
                sucess+=1
                break
    print('sucess:',sucess)
if __name__ == "__main__":
    main()    