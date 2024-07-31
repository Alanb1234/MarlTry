# Following the youtube tutorial:
# Deep Q Learning is Simple with PyTorch | Full Tutorial 2020

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # For adam optimizer
import numpy as np


import gym



# Just using a replay network 

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__() # Call the constructor of the parent class
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # Want to unpack the input_dims tuple
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions) # Output layer
        self.optimizer = optim.Adam(self.parameters(), lr=lr) # Adam optimizer
        self.loss = nn.MSELoss() # Mean squared error loss
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # Check if cuda is available
        self.to(self.device) 
        
    # Handing the forward propogation
    def forward(self, state):
        x = F.relu(self.fc1(state)) # Pass the state through the first layer
        x = F.relu(self.fc2(x)) # Pass the output of the first layer through the second layer
        actions = self.fc3(x) # Pass the output of the second layer through the output layer
        
        return actions
    


class Agent():
    # gamma = discount factor, epsilon = exploration factor, lr = learning rate, input_dims = input dimensions, batch_size = batch size, 
    # n_actions = number of actions, max_mem_size = maximum memory size, eps_end = epsilon end, eps_dec = epsilon decrease
    def __init__(self,gamma,epsilon,lr, input_dims, bathc_size, n_actions,
                 max_mem_size = 100000, eps_end = 0.01, eps_dec = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = bathc_size
        self.n_actions = n_actions
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        
        self.Q_eval = DeepQNetwork(self.lr, self.input_dims, 256, 256, self.n_actions)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size # Replace the old memory
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
            
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32) # To perform proper array slicing
        
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0 # values of terminal states are 0
        
        q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0] # Bellman equation
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end


    def save_model(self):
        T.save(self.Q_eval.state_dict(), 'model')   

    def load_model(self):
        self.Q_eval.load_state_dict(T.load('model'))
    

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    