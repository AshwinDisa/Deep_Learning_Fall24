import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Data Preparation
transform_train = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform_train)

# Use only 1000 samples from the training set
subset_indices = list(range(1000))
trainset_subset = torch.utils.data.Subset(trainset, subset_indices)
trainloader = torch.utils.data.DataLoader(
    trainset_subset, batch_size=64, shuffle=True)

# Define the FCNN
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

def get_parameters_vector(net):
    return torch.cat([param.view(-1) for param in net.parameters()])

# Training Loop with Trajectory Recording
def train_and_record(net, optimizer, criterion, num_epochs, trainloader):
    parameter_trajectory = []
    loss_trajectory = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Compute average loss for the epoch
        avg_loss = running_loss / len(trainloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Save parameters and loss
        params = get_parameters_vector(net).detach().clone()
        parameter_trajectory.append(params)
        loss_trajectory.append(avg_loss)

    return parameter_trajectory, loss_trajectory

# Number of runs
num_runs = 2
all_parameter_trajectories = []
all_loss_trajectories = []

for run in range(num_runs):
    print(f'\nStarting training run {run + 1}/{num_runs}')

    # Initialize the network and optimizer
    net = FCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # Train the network and record trajectories
    param_traj, loss_traj = train_and_record(net, optimizer, criterion, num_epochs=25, trainloader=trainloader)

    # Store the trajectories
    all_parameter_trajectories.append(param_traj)
    all_loss_trajectories.append(loss_traj)

all_params = []
for traj in all_parameter_trajectories:
    for params in traj:
        all_params.append(params.numpy())

all_params = np.array(all_params)

# Perform PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_params)

num_points = 25
x = np.linspace(-2, 2, num_points) 
y = np.linspace(-2, 2, num_points)
X, Y = np.meshgrid(x, y)
grid_points = np.vstack([X.ravel(), Y.ravel()]).T

# Project grid points back into full parameter space using inverse transform
full_param_space_points = pca.inverse_transform(grid_points)

# Function to load parameters into the neural network
def load_parameters_into_network(net, params):
    current_idx = 0
    for param_tensor in net.parameters():
        num_elements = param_tensor.numel()
        reshaped_params = params[current_idx:current_idx + num_elements].reshape(param_tensor.shape)
        param_tensor.data.copy_(torch.tensor(reshaped_params))
        current_idx += num_elements

# Function to compute loss on a fixed batch of data (1000 examples) from trainloader
def compute_loss_for_params(net, criterion, data_loader):
    net.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

# Compute loss values for each point on the grid in full parameter space
loss_values = []
for params in full_param_space_points:
    load_parameters_into_network(net, params)  
    loss_value = compute_loss_for_params(net, criterion, trainloader)
    loss_values.append(loss_value)

loss_values = np.array(loss_values).reshape(X.shape) 

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(X, Y, loss_values, cmap='viridis', alpha=0.6)

colors = ['blue', 'red', 'green', 'orange']
markers = ['o', '^', 's', 'd']

start_idx = 0

for run_idx, (traj, loss_traj) in enumerate(zip(all_parameter_trajectories, all_loss_trajectories)):
    end_idx = start_idx + len(traj)

    xs = pca_result[start_idx:end_idx, 0]
    ys = pca_result[start_idx:end_idx, 1]
    zs = loss_traj

    ax.plot(xs, ys, zs,
            label=f'Run {run_idx + 1}',
            marker=markers[run_idx % len(markers)],
            color=colors[run_idx % len(colors)],
            linestyle='-')

    start_idx = end_idx

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Cross-Entropy Loss')
ax.set_title('Loss Landscape with SGD Trajectories')

ax.legend()

ax.view_init(elev=30, azim=45)

plt.show()