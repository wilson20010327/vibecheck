import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ENV.env import RLEnv
import pandas as pd
import ast,os
import numpy as np
from collections import deque
model_name='model_try0205.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def save_trajectory(trajectory, episode_num, path):
    """Save trajectory to a JSON file."""
    file_path = os.path.join(path, f"trajectory_episode_{episode_num}.csv")
    df = pd.DataFrame(trajectory)  # Convert trajectory to a DataFrame
    df.to_csv(file_path, index=False)  # Save to Excel
    print(f"Trajectory for episode {episode_num} saved to {file_path}")    
def print_original_plot(path):
  material=path
  filename="/all_result.csv"
  df=pd.read_csv(material+filename)
  df=df.dropna(axis=0) # drop the unconsistent data long
  df=df.drop(list(df)[0],axis=1)
  df['state']=df['state'].apply(lambda x: np.array(ast.literal_eval(x)))
  return df
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = F.relu(self.fc2(x))  # Apply ReLU activation
        x = F.softmax(self.fc3(x), dim=1)  # Apply softmax to output layer
        return x
    
class MLP_Agent():
    def __init__(self, input_size, hidden_size, output_size,result_path, lr=0.001):
        self.model = MLPModel(input_size, hidden_size, output_size)  # Create the model
        self.model=self.model.to('cuda')
        self.criterion = nn.CrossEntropyLoss()  # Define the loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Define the optimizer
        self.result_path = result_path
    def extract_input(self,input):
        output=[]
        for i in input:
            temp=[]
            for j in i:
                temp.append(j)
            output.append(temp)
        return output 
    def extract_target(self,input):
        output=[]
        for i in input:
            output.append(i)
        return output        
    def train(self, inputs,targets,epochs,batch_size):
        for epoch in range(epochs):
        # Shuffle data if needed
            permutation = torch.randperm(len(inputs))
            epoch_loss = 0.0
            
            for i in range(0, len(inputs), batch_size):
                # Prepare the mini-batch
                indices = permutation[i:i + batch_size]
                batch_inputs = self.extract_input(inputs.iloc[indices])
                batch_targets = self.extract_target(targets.iloc[indices])
                # print(i)
                # Forward pass
                outputs = self.forward(batch_inputs)
                # print('outputs')
                # Compute the loss
                batch_targets=torch.tensor(batch_targets).long().to("cuda")
                loss = self.criterion(outputs, batch_targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Accumulate loss
                epoch_loss += loss.item()
            
            # Print epoch progress
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        return epoch_loss
    def forward(self, inputs):
        inputs=torch.tensor(inputs).float().to("cuda")
        return self.model(inputs)
    def act(self, inputs):
        outputs = self.forward(inputs)
        _, predicted = torch.max(outputs, 1)
        return predicted.to("cpu")
    def save(self):
        torch.save(self.model.state_dict(), self.result_path+model_name)
        print('model saved')
    def load_model(self):
        self.model.load_state_dict(torch.load(self.result_path+model_name))
        self.model.eval()
        print('model loaded')

class Observation():
    def __init__(self):
        self.observation=deque([[0,0,0] for i in range(10)],maxlen=10)
        self.observation_size=30
    def append(self,observation):
        temp=[0,0,0]
        temp[observation]=1
        self.observation.append(temp)
    def get_observation(self):
        return np.array(self.observation).flatten()
    def reset(self):
        self.observation=deque([[0,0,0] for i in range(10)],maxlen=10)
        return
if __name__ == '__main__':
   
    obs=Observation()
    # interact with env
    # env=RLEnv(random_flag=True,perfect=False)
    result_path="./policy/" # path to the model parameters
    mlp_agant=MLP_Agent(input_size=obs.observation_size, output_size=5, hidden_size=128,result_path=result_path)
    mlp_agant.load_model()
    # sucess=0
    # for j in range(1000):
    trajectory = [] 
    # tot_reward = 0
    # robot.reset()
    pose=pose_model_detetion()
    
    if(pose ==2 ):
        return
    
    obs.append(pose)
    
    current = obs.get_observation() # format =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]"
    # groundtruth=env.euler
    # record=current
    
    # current=mlp_agant.extract_input([current])
    action= mlp_agant.act(current) # [0]   action=[0]   action[0]=0
    # next, reward, done, info = env.step(action[0])
    robot_move(action[0])
    
    # tot_reward += reward

    # trajectory.append({
    #     'step': i,
    #     'real_state': groundtruth,
    #     'state': record.tolist() if isinstance(record, np.ndarray) else record,
    #     'action': int(action),
    #     'reward': reward,
    #     'tot_reward': tot_reward,
    #     'next_state': next.tolist() if isinstance(next, np.ndarray) else next,
    #     'done': done,
    # })
        

   