import numpy as np
import random
from collections import deque
import transforms3d
from transforms3d.euler import mat2euler
import math
from pynput import keyboard
class RLEnv:
    def __init__(self,random_flag=False):
        self.current_state = [0,0] #[ x_count, z_count]
        self.euler=[0,0,0] # [x,y,z] unit:degree
        self.transformMatrix = np.eye(3,3)
        self.labellist=['diagonal','one_line','in_hole']
        self.action_label = ['z','n','x','m']
        self.predifction_confusion_matrix = np.array([
            [29,1,0],
            [3,27,0],
            [0, 0,1]])
        self.discrete_size=10
        self.observation=deque([[0,0,0] for i in range(10)],maxlen=10)
        self.observation_size = 30
        self.action_size = 4
        self.random_flag=random_flag
        if self.random_flag:
            self.euler=self.random_euler()
    def random_euler(self):
        step_size = 45 / self.discrete_size

        # Generate random x and z values in multiples of step_size within [-180, 180]
        x=random.randint(-180/step_size, 180/step_size)*step_size
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
        if euler[2]!=45:
            return 0 # diagonal
        elif euler[0]!= 45:
            return 1 # one_line
        else:
            return 2 # in_hole
    def reset(self):
        self.current_state = [0,0] #[ x_count, z_count]
        self.euler=[0,0,0]
        if self.random_flag:
            self.euler=self.random_euler()
        self.observation=deque([[0,0,0] for i in range(10)],maxlen=10)
        return
    def action2state(self,action):
        next_state = self.current_state.copy()
        next_marix=self.transformMatrix.copy()
        next_euler=self.euler.copy()
        R0=0
        if action=='z':
            next_state[1]+=1
            next_euler[2]+=45/self.discrete_size
            # R0 = transforms3d.axangles.axangle2mat([0,0,1], (45)*math.pi/180/self.discrete_size)
            # next_marix=R0@next_marix
        elif action=='n':
            next_state[1]-=1
            next_euler[2]-=45/self.discrete_size
            # R0 = transforms3d.axangles.axangle2mat([0,0,1], (-45)*math.pi/180/self.discrete_size)
            # next_marix=R0@next_marix
        elif action=='x':
            next_state[0]+=1
            next_euler[0]+=45/self.discrete_size
            # R0 = transforms3d.axangles.axangle2mat([1,0,0], (45)*math.pi/180/self.discrete_size)
            # next_marix=R0@next_marix
        elif action=='m':
            next_state[0]-=1
            next_euler[0]-=45/self.discrete_size
            # R0 = transforms3d.axangles.axangle2mat([1,0,0], (-45)*math.pi/180/self.discrete_size)
            # next_marix=R0@next_marix
        next_euler = [self.normalize_angle(angle) for angle in next_euler]
        # next_state[0] = min(max(0,next_state[0]),self.discrete_size)
        # next_state[1] = min(max(0,next_state[1]),self.discrete_size)
        # print('R0:\n',R0)
        # print('next_marix:\n',next_marix)

        # Convert from radians to degrees
        # Extract angles back
        # extracted_theta_z = np.arctan2(next_marix[1, 0], next_marix[0, 0])
        # extracted_theta_x = np.arctan2(next_marix[2, 1], next_marix[2, 2])

        # Convert back to degrees
        # extracted_theta_z_deg = np.degrees(extracted_theta_z)
        # extracted_theta_x_deg = np.degrees(extracted_theta_x)
        # print('extracted_theta_z_deg:',extracted_theta_z_deg)
        # print('extracted_theta_x_deg:',extracted_theta_x_deg)
        # self.transformMatrix=next_marix
        return next_state,next_euler
    def get_model_prediction(self,groundtruth_label):
        dist=self.predifction_confusion_matrix[groundtruth_label]  
        selected_index = random.choices(range(len(dist)), weights=dist)[0]
        return selected_index 
    def get_gt_action(self,euler):

        if euler[2] <45:
            return 'z'
        if euler[2] >45:
            return 'n'        
        if euler[0] <45:
            return 'x'
        if euler[0] >45:
            return 'm'

        return 'f' # in_hole
    def reward_function(self,action):
        gt_action = self.get_gt_action(self.euler)
        # print('gt_action:',gt_action)
        return 1 if gt_action == self.action_label[action] else -1        
    def step(self, action):
        next_state,next_euler = self.action2state(self.action_label[action])
        groundtruth_label = self.get_ground_truth_label(self.euler)
        # print('previous euler:',self.euler)
        # print('groundtruth_label:',groundtruth_label)
        
        temp=[0,0,0]
        sampled_label = self.get_model_prediction(groundtruth_label)  # apply observation model based-on model accuracy
        temp[sampled_label]=1
        self.observation.append(temp)
        
        reward = self.reward_function(action)
        self.current_state = next_state
        self.euler=next_euler
        # print('after action euler:',self.euler)
        done = groundtruth_label == 2
        if done:
            reward = 10
            return self.get_observation(), reward, 1, {}
        return self.get_observation(), reward, 0, {}
if __name__=='__main__':
    env = RLEnv()
    tot_reward = 0
    action_map = {'z': 0, 'n': 1, 'x': 2, 'm': 3}
    def on_press(key):
        global tot_reward,action_map,env
        try:
            if hasattr(key, 'char') and key.char in action_map:
                action = action_map[key.char]
                print('Action:', action)
                observation, reward, done, info = env.step(action)
                tot_reward += reward

                print(f"Observation: \n{observation}, \nReward: {reward}, \nTotal Reward: {tot_reward}, \nDone: {done}")

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