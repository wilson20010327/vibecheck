import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Define the MLP model
import pandas as pd
import ast,os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import KernelPCA
from sklearn.neural_network import MLPClassifier
import numpy as np
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
                batch_targets=torch.tensor(batch_targets).long()
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
        inputs=torch.tensor(inputs).float()
        return self.model(inputs)
    def act(self, inputs):
        outputs = self.forward(inputs)
        _, predicted = torch.max(outputs, 1)
        return predicted
    def save(self):
        torch.save(self.model.state_dict(), self.result_path+'model.pth')
        print('model saved')
    def load_model(self):
        self.model.load_state_dict(torch.load(self.result_path+'model.pth'))
        self.model.eval()
        print('model loaded')
        
if __name__ == '__main__':
    # Define the model hyperparameters
    path="./result/IL/training/1/"
    folder_list = os.listdir(path)

    df_temp=print_original_plot(path=path)

    X=df_temp["state"]
    y=df_temp["action"].astype(int)
    
    input_size=len(X.iloc[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlp_agant=MLP_Agent(input_size=input_size, output_size=5, hidden_size=128,result_path="./result/IL/training/")

    mlp_agant.train(X_train, y_train,epochs=1000,batch_size=100)
    mlp_agant.save()