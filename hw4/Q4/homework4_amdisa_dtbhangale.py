import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNNPolicyNet(nn.Module):

    def __init__(self, num_actions):

        super(CNNPolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(4608, 128)                 # 4608 = 128 * 6 * 6              
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)    

        return policy_logits, value
    
    def predict(self, x, device):

        # x to tensor and float32
        x = torch.tensor(x, dtype=torch.float32).to(device)
        x = x.permute(0, 3, 1, 2)    # shape: (1, 4, 84, 84)
        policy_logits, value = self.forward(x)
        return F.softmax(policy_logits, dim=-1), value        
    
def load_expert_data(observation_file, action_file):
    observations = torch.load(observation_file)        # shape: (15261, 84, 84, 4)
    actions = torch.load(action_file)                  # shape: (15261,) 

    observations = observations.permute(0, 3, 1, 2)    # shape: (15261, 4, 84, 84) 
    return observations, actions

def train(model, dataloader, optimizer, criterion_policy, criterion_value, device):

    epoch_loss = 0
    model.train()
    for batch in dataloader:
        obs, actions = batch
        obs, actions = obs.to(device), actions.to(device)
        policy_logits, value = model(obs)
        
        # compute policy_loss (cross-entropy) and value_loss (MSE)
        policy_loss = criterion_policy(policy_logits, actions)
        value_loss = criterion_value(value, torch.zeros_like(value))    

        # compute total loss
        loss = policy_loss + value_loss

        epoch_loss += loss.item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return epoch_loss / len(dataloader)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    observations, actions = load_expert_data("pong_observations.pt", "pong_actions.pt")
    dataset = TensorDataset(observations, actions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_actions = 6
    model = CNNPolicyNet(num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    # train model
    epochs = 100
    for epoch in range(epochs):
        loss = train(model, dataloader, optimizer, criterion_policy, criterion_value, device)
        print(f"Loss: {loss}, Epoch {epoch+1}/{epochs} complete")

    # save model
    torch.save(model.state_dict(), "model.cpt")    

