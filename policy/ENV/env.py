import numpy as np
import random
from collections import deque
import transforms3d
from transforms3d.euler import mat2euler
import math
from pynput import keyboard
class RLEnv:
    def __init__(self,random_flag=False,perfect=False):
        self.current_state = [0,0] #[ x_count, z_count]
        self.euler=[0,0,0] # [x,y,z] unit:degree
        self.transformMatrix = np.eye(3,3)
        self.labellist=['diagonal','one_line','in_hole']
        self.action_label = ['z','x','f']
        self.predifction_confusion_matrix = np.array([
            [28,1,0],
            [28,1,0],
            [28,1,0],
            [27,2,0],
            [23,6,0],
            [25,4,0],
            [24,5,0],
            [20,9,0],
            [20,9,0],
            [14,15,0],
            [1,19,0],
            [5,15,0],
            [2,18,0],
            [1,19,0],
            [1,19,0],
            [2,18,0],
            [1,19,0],
            [2,18,0],
            [1,18,1],
            [0,12,8],
            [0, 5,195]])
        if perfect:
            self.predifction_confusion_matrix = np.array([
                [1,0,0],
                [1,0,0],
                [1,0,0],
                [1,0,0],
                [1,0,0],
                [1,0,0],
                [1,0,0],
                [1,0,0],
                [1,0,0],
                [1,0,0],
                [0,1,0],
                [0,1,0],
                [0,1,0],
                [0,1,0],
                [0,1,0],
                [0,1,0],
                [0,1,0],
                [0,1,0],
                [0,1,0],
                [0,1,0],
                [0, 0,1]])
        self.discrete_size=10
        self.observation=deque([[0,0,0] for i in range(10)],maxlen=10)
        self.observation_size = 30
        self.action_size = 3
        self.random_flag=random_flag
        if self.random_flag:
            self.euler=self.random_euler()
        self.model_detetion()
    def random_euler(self):
        step_size = 45 / self.discrete_size

        # Generate random x and z values in multiples of step_size within [-180, 180]
        x=random.randint(-1, 8)*step_size
        z=random.randint(-180/step_size, 180/step_size)*step_size

        # y is fixed to 0
        y = 0
        return [x, y, z]
    def get_observation(self):
        temp=[]
        for i in range(len(self.observation)):
            temp.append(self.observation[i][0])
            temp.append(self.observation[i][1])
            temp.append(self.observation[i][2])
        return np.array(temp).astype('float32')
    def normalize_angle(self, angle):
        """Normalize an angle to the range -180 to 180."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle       
    def get_ground_truth_label(self,euler):
        if (euler[2]==45 or euler[2]==-45 or euler[2]==135 or euler[2]==-135) and euler[0]== 45:
            return len(self.predifction_confusion_matrix)-1 # in_hole
        elif euler[2]==45 or euler[2]==-45 or euler[2]==135 or euler[2]==-135:
            temp=int(abs(euler[0])/4.5)+10
            return temp # one_line
        else:
            temp=abs(euler[2])
            if (temp>90):
                temp=180-temp
            if(temp>45):
                temp=90-temp
            temp=int(temp/4.5)
            return temp # in_hole
    def reset(self):
        self.current_state = [0,0] #[ x_count, z_count]
        self.euler=[0,0,0]
        if self.random_flag:
            self.euler=self.random_euler()
        self.observation=deque([[0,0,0] for i in range(10)],maxlen=10)
        
        self.model_detetion()
        return
    def action2state(self,action):
        next_state = self.current_state.copy()
        next_euler=self.euler.copy()
        if action=='z':
            next_state[1]+=1
            next_euler[2]+=45/self.discrete_size

        elif action=='x':
            if(next_euler[0]==45): return next_state,next_euler
            next_state[0]+=1
            next_euler[0]+=45/self.discrete_size

        next_euler = [self.normalize_angle(angle) for angle in next_euler]
       
        return next_state,next_euler
    def get_model_prediction(self,groundtruth_label):
        dist=self.predifction_confusion_matrix[groundtruth_label]  
        selected_index = random.choices(range(len(dist)), weights=dist)[0]
        return selected_index 
    def get_gt_action(self,euler):
        if euler[2]!=45 and euler[2]!=-45 and euler[2]!=135 and euler[2]!=-135:
            return 'z'
        elif euler[0]!= 45:
            return 'x'
        # if euler[2] <45:
        #     return 'z'
        # if euler[2] >45:
        #     return 'n'        
        # if euler[0] <45:
        #     return 'x'
        # if euler[0] >45:
        #     return 'm'

        return 'f' # in_hole
    def reward_function(self,action):
        gt_action = self.get_gt_action(self.euler)
        # print('gt_action:',gt_action)
        return 1 if gt_action == self.action_label[action] else -1
    def model_detetion(self):
        groundtruth_label = self.get_ground_truth_label(self.euler)
        # print('previous euler:',self.euler)
        # print('groundtruth_label:',groundtruth_label)
        
        temp=[0,0,0]
        sampled_label = self.get_model_prediction(groundtruth_label)  # apply observation model based-on model accuracy
        temp[sampled_label]=1
        self.observation.append(temp)   
        return groundtruth_label     
    def step(self, action):
        next_state,next_euler = self.action2state(self.action_label[action])
        reward = self.reward_function(action)
        self.current_state = next_state
        self.euler=next_euler
        
        groundtruth_label=self.model_detetion()
        # print('after action euler:',self.euler)
        done = groundtruth_label == len(self.predifction_confusion_matrix)-1
        if done:
            reward = 100
            return self.get_observation(), reward, 1, {}
        return self.get_observation(), reward, 0, {}
if __name__=='__main__':
    env = RLEnv(perfect=True,random_flag=True)
    tot_reward = 0
    action_map = {'z': 0, 'x': 1, 'f': 2}
    def on_press(key):
        global tot_reward,action_map,env
        try:
            if hasattr(key, 'char') and key.char in action_map:
                action = action_map[key.char]
                print('Action:', action)
                observation, reward, done, info = env.step(action)
                tot_reward += reward

                print(f"state :{env.euler} \nObservation: \n{observation}, \nReward: {reward}, \nTotal Reward: {tot_reward}, \nDone: {done}")

                if done:
                    env.reset()
                    tot_reward = 0
                    print('Environment reset.')
                    # return False  # Exit listener

                print('-----------------------------')
        except Exception as e:
            print(f"Error: {e}")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()  # Wait for a key press
    print('-----------------------------')