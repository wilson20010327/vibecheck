import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


from model.ReplayMemory import ReplayMemory 
from model.utils import soft_update_target_network, hard_update_target_network

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self,result_path,test=False ,state_size=10, action_size=4, hidden_size=32, lr=0.01, gamma=0.99, epsilon=0.99,tau=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau=tau
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.test=test
        self.result_path = result_path
        if self.test:
            self.Q = DQN(state_size, action_size, hidden_size)
            self.epsilon = 0
            self.load_model()
        if not self.test:
            self.Q = DQN(state_size, action_size, hidden_size)
            self.target_Q=DQN(state_size, action_size, hidden_size)
            hard_update_target_network(self.target_Q,self.Q )
            self.target_Q.eval()
            self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)
            self.criterion = nn.MSELoss()

            self.memory = ReplayMemory(10000)
            
     
    def add_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
    def step(self,state, action, reward, next_state, done):
        self.add_memory(state, action, reward, next_state, done)
        loss=None
        if len(self.memory) > 1000:
            loss=self.train()
            # print('loss: ',loss)    
        return loss    
    def act(self, state):
        state=torch.tensor(state)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if not self.test and torch.rand(1).item() < self.epsilon:
            # print('random')
            action=torch.randint(0, self.action_size, (1,))[0]
            
            return action , False 
        else:
            # print('greedy')
            with torch.no_grad():
                action=torch.argmax(self.Q(state)).item()
                return action, True 

    def train(self ):
        state, action, reward, next_state, done = self.memory.sample(100)
        state=torch.tensor(state)
        action=torch.tensor(action).reshape(-1,1)
        reward=torch.tensor(reward).squeeze()
        next_state=torch.tensor(next_state)
        done=torch.tensor(done)
        # print('state:',state)
        # print('action:',action)
        # print('reward:',reward)
        # print('next_state:',next_state)
        # print('done:',done)
        with torch.no_grad():
            pred_Q_a = self.target_Q(next_state)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()
            
            # Compute the TD error
            target = reward + (1 - done) * self.gamma * Qprime
        q_values = self.Q.forward(state)
        
        y_predicted = q_values.gather(1, action).squeeze(1)  # Select the corresponding action q value
        loss = self.criterion(y_predicted, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        soft_update_target_network(self.target_Q,self.Q,self.tau)
        
        return loss.item()
    def save(self):
        torch.save(self.Q.state_dict(), self.result_path+'model.pth')
        print('model saved')
    def load_model(self):
        self.Q.load_state_dict(torch.load(self.result_path+'model.pth'))
        self.Q.eval()
        print('model loaded')