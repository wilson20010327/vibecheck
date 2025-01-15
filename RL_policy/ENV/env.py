import numpy as np
import random
from collections import deque
class RLEnv:
    def __init__(self):
        self.current_state = [0,0] #[ x_count, z_count]
        self.labellist=['diagonal','one_line','in_hole']
        self.action_label = ['z','n','x','m']
        self.predifction_confusion_matrix = np.array([[29,1,0],[3,27,0],[0,0,1]])
        self.discrete_size=10
        self.observation=deque([-1 for i in range(10)],maxlen=10)
        
    def get_observation(self):
        temp=[]
        for i in range(len(self.observation)):
            temp.append(self.observation[i])
        return np.array(temp).astype('float32')
           
    def get_ground_truth_label(self,current_state):
        if current_state[1]<self.discrete_size:
            return 0 # diagonal
        elif current_state[0]<self.discrete_size:
            return 1 # one_line
        else:
            return 2 # in_hole
    def reset(self):
        self.current_state = [0,0] #[ x_count, z_count]
        self.observation=deque([-1 for i in range(10)],maxlen=10)
        return
    def action2state(self,action):
        next_state = self.current_state.copy()
        if action=='z':
            next_state[1]+=1
        elif action=='n':
            next_state[1]-=1
        elif action=='x':
            next_state[0]+=1
        elif action=='m':
            next_state[0]-=1
        next_state[0] = min(max(0,next_state[0]),self.discrete_size)
        next_state[1] = min(max(0,next_state[1]),self.discrete_size)
        return next_state
    def get_model_prediction(self,groundtruth_label):
        dist=self.predifction_confusion_matrix[groundtruth_label]  
        selected_index = random.choices(range(len(dist)), weights=dist)[0]
        return selected_index 
    def get_gt_action(self,current_state):
        if (current_state[0] ==0 and current_state[1]<self.discrete_size):
            return 'z'
        if (current_state[0] !=0 and current_state[1]<self.discrete_size):
            return 'm'
        if (current_state[0] <self.discrete_size and current_state[1]==self.discrete_size):
            return 'x'
        if (current_state[1]>self.discrete_size):
            return 'n'
        if (current_state[0] >self.discrete_size):
            return 'm'
        return 'f' # in_hole
    def reward_function(self,action):
        gt_action = self.get_gt_action(self.current_state)
        return 1 if gt_action == action else -1        
    def step(self, action):
        next_state = self.action2state(self.action_label[action])
        groundtruth_label = self.get_ground_truth_label(self.current_state)

        sampled_label = self.get_model_prediction(groundtruth_label)  # apply observation model based-on model accuracy
        self.observation.append(sampled_label)
        
        reward = self.reward_function(action)
        self.current_state = next_state
        done = groundtruth_label == 2
        if done:
            reward = 10
            return self.get_observation(), reward, 1, {}
        return self.get_observation(), reward, 0, {}
if __name__=='__main__':
    env = RLEnv()
    tot_reward = 0
    trajectory = ['z','z','z','z','z','z','z','z','z','z','x','x','x','x','x','x','x','x','x','x','n']
    for i in trajectory:
        action = i
        print('current_state:',env.current_state)
        print('action:',action)
        observation, reward, done, info = env.step(action)
        tot_reward+=reward
        print('observation:',observation)
        print('reward:',reward)
        print('tot_reward:',tot_reward)
        print('done:',done)
        if done:
            env.reset()
            tot_reward = 0
            print('reset')
            break
        print('-----------------------------')