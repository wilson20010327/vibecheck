import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ENV.env import RLEnv
import pandas as pd
import ast,os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
model_name='model_try-limit-init2016.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
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
        
if __name__ == '__main__':
    # Define the model hyperparameters
    # path="./result/IL/training/new-confusion-outdistribution-limit-init-0216/"
    # folder_list = os.listdir(path)

    # df_temp=print_original_plot(path=path)

    # X=df_temp["state"]
    # y=df_temp["action"].astype(int)
    
    # input_size=len(X.iloc[0])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # mlp_agant=MLP_Agent(input_size=input_size, output_size=5, hidden_size=128,result_path="./result/IL/training/")
    # # mlp_agant.load_model()
    # mlp_agant.train(X_train, y_train,epochs=100,batch_size=100)
    # mlp_agant.save()
    
    # print("Classification Report:")
    # X_test=mlp_agant.extract_input(X_test)
    # print(classification_report(y_test, mlp_agant.act(X_test)))

    
    # interact with env
    env=RLEnv(random_flag=True,perfect=False)
    mlp_agant=MLP_Agent(input_size=env.observation_size, output_size=5, hidden_size=128,result_path="./result/IL/training/")
    mlp_agant.load_model()
    sucess=0
    for j in range(1000):
        trajectory = [] 
        tot_reward = 0
        env.reset()
        
        for i in range(50):
            current = env.get_observation()
            # print(current)
            groundtruth=env.euler
            record=current
            current=mlp_agant.extract_input([current])
            
            action= mlp_agant.act(current)
            # print('action:',action)
            next, reward, done, info = env.step(action[0])
            tot_reward += reward
            # print('ground true:',env.euler)
            # print('next:',next)
            # print('reward:',reward)
            # print('tot_reward:',tot_reward)
            # print('done:',done)
            trajectory.append({
                'step': i,
                'real_state': groundtruth,
                'state': record.tolist() if isinstance(record, np.ndarray) else record,
                'action': int(action),
                'reward': reward,
                'tot_reward': tot_reward,
                'next_state': next.tolist() if isinstance(next, np.ndarray) else next,
                'done': done,
            })
            if done:
                sucess+=1
                # print('sucess:',sucess)
                break
        # save_trajectory(trajectory, j, mlp_agant.result_path+"/../test/newconfustion/")
    print('sucess:',sucess)