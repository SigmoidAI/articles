import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import torch
import torch.nn as nn


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine implementation to extract the features from the input data in an unsupervised manner
    """
    def __init__(self, visible_units, hidden_units):
        """
        Initialize the Restricted Boltzmann Machine

        Args:
            visible_units: the number of visible units
            hidden_units: the number of hidden units
        """
        super(RBM, self).__init__()
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.weights = nn.Parameter(torch.empty(hidden_units, visible_units))  # weight matrix
        nn.init.xavier_uniform_(self.weights)  # weights initialization
        self.visible_bias = nn.Parameter(torch.zeros(visible_units))  # bias for the visible units
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_units))  # bias for the hidden units

    def sample_h(self, v):
        """
        Computes the probability of hidden units given the visible units
        Args:
            v: activations for visible layers
        Returns:
            The probability of hidden layer activations
        """
        h_prob = torch.sigmoid(torch.matmul(v, self.weights.t()) + self.hidden_bias)
        return h_prob

    def sample_v(self, h):
        """
        Computes the probability of visible units given the hidden units
        Args:
            h: activations for hidden layers
        Returns:
            The probability of visible layer activations
        """
        v_prob = torch.sigmoid(torch.matmul(h, self.weights) + self.visible_bias)
        return v_prob

    def pretrain(self, rbm_train_data, rbm_val_data, rbm_batch_size, epochs=10, lr=0.001):
        """
        Pretrain the Restricted Boltzmann Machine using unsupervised learning, minimizing the reconstruction
        errors using Contrastive Divergence

        Args:
            rbm_train_data: training data
            rbm_val_data: validation data
            rbm_batch_size: the batch size for training
            epochs: number of training epochs
            lr: learning rate

        Returns:
            The lists of training and validation losses for each epoch
        """
        num_samples = rbm_train_data.size(0)
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            epoch_loss = 0
            indices = torch.randperm(num_samples)
            data = rbm_train_data[indices]

            for i in range(0, num_samples, rbm_batch_size):
                batch = data[i:i + rbm_batch_size].to(next(self.parameters()).device)
                loss = self.train_rbm(batch, lr)
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= num_samples
            train_losses.append(epoch_loss)

            # visualize the reconstructed images for each 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"Reconstruction on epoch {epoch + 1}:")
                self.visualize_reconstructions(data)
            with torch.no_grad():
                val_loss = self.evaluate(rbm_val_data)
                val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        return train_losses, val_losses

    def train_rbm(self, v, lr=0.01):
        """
        Perform one training step of Contrastive Divergence

        Args:
            v: the input batch of visible layer data, shape (batch_size, visible_units)
            lr: learning rate for the parameter updates

        Returns:
            Reconstruction loss for the input batch
        """
        h_prob = self.sample_h(v)
        v_reconstructed = self.sample_v(h_prob)
        h_prob_reconstructed = self.sample_h(v_reconstructed)

        positive_grad = torch.matmul(v.t(), h_prob)
        negative_grad = torch.matmul(v_reconstructed.t(), h_prob_reconstructed)

        with torch.no_grad():
            self.weights += lr * (positive_grad - negative_grad).t()
            self.visible_bias += lr * torch.sum(v - v_reconstructed, dim=0)
            self.hidden_bias += lr * torch.sum(h_prob - h_prob_reconstructed, dim=0)

        return torch.mean((v - v_reconstructed) ** 2)

    def evaluate(self, data):
        """
        Compute the reconstruction error on the given dataset to evaluate the RBM performance

        Args:
            data: the dataset to evaluate, shape (num_samples, visible_units)

        Returns:
            Mean squared reconstruction error
        """
        with torch.no_grad():
            v_reconstructed = self.sample_v(self.sample_h(data))
            return torch.mean((data - v_reconstructed) ** 2).item()

    @torch.no_grad()
    def extract_features(self, data):
        """
        Extract features from the hidden layer of the RBM

        Args:
            data: input data to extract features from, shape (num_samples, visible_units)

        Returns:
            The activations for the hidden layers, shape (num_samples, hidden_units)
        """
        h_activations = self.sample_h(data)
        return h_activations

    @torch.no_grad()
    def visualize_reconstructions(self, data, num_samples=16):
        """
        Visualize original and reconstructed images for the evaluated samples

        Args:
            data: the input dataset to reconstruct.
            num_samples (default: 16): the number of samples to visualize
        """
        v = data[:num_samples]
        v_reconstructed = self.sample_v(self.sample_h(v))
        fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 4))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(v[i].view(28, 28).cpu(), cmap="gray")
            ax.axis("off")
        plt.suptitle("Original images:")
        plt.show()

        fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 4))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(v_reconstructed[i].view(28, 28).cpu(), cmap="gray")
            ax.axis("off")
        plt.suptitle("Reconstructed images:")
        plt.show()


class DBN(nn.Module):
    """
    Deep Belief Network implementation to combine multiple Restricted Boltzmann Machines and adding supervised training
    for classification task
    """
    def __init__(self, rbm_layers, output_classes):
        """
        Initialize the Deep Belief Network

        Args:
            rbm_layers: a list of pretrained RBMs stacked together
            output_classes: the number of output classes for the classification task
        """
        super(DBN, self).__init__()
        self.rbms = nn.ModuleList(rbm_layers)
        self.classifier = nn.Sequential(
            nn.Linear(rbm_layers[-1].hidden_units, output_classes),  # fully connected layer
            nn.Softmax(dim=1),  # Softmax for classification
        )

    def forward(self, x):
        """
        Perform a forward pass through the DBN

        Args:
            x: the input tensor, shape (batch_size, visible_units)

        Returns:
            The logits for each class, shape (batch_size, output_classes)
        """
        for rbm in self.rbms:
            x = torch.sigmoid(torch.matmul(x, rbm.weights.t()) + rbm.hidden_bias)
        return self.classifier(x)


def train_dbn(dbn, dbn_train_data, dbn_train_labels, dbn_val_data, dbn_val_labels, epochs=25, lr=0.05):
    """
    Train the Deep Belief Network (DBN) using supervised learning

    Args:
        dbn: the Deep Belief Network model to train
        dbn_train_data: the training data
        dbn_train_labels: the training data labels
        dbn_val_data: the validation data
        dbn_val_labels: the validation labels
        epochs (default: 15): the number of training epochs
        lr (default: 0.05): learning rate for the Adam optimizer

    Returns:
        The lists of training and validation losses per epoch
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dbn.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        dbn.train()
        optimizer.zero_grad()
        outputs = dbn(dbn_train_data)
        train_loss = criterion(outputs, dbn_train_labels)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        dbn.eval()
        with torch.no_grad():
            val_outputs = dbn(dbn_val_data)
            val_loss = criterion(val_outputs, dbn_val_labels)
            val_losses.append(val_loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


def evaluate(dbn_model, data, labels):
    """
    Evaluate the trained DBN on a test dataset

    Args:
        dbn_model: the trained Deep Belief Network model
        data: the test data, shape (num_samples, visible_units)
        labels: the test data labels, shape (num_samples)
    """
    with torch.no_grad():
        outputs = dbn_model(data)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).sum().item() / len(labels)

    print(f"Accuracy: {accuracy:.2f}")


def plot_losses(train_losses, val_losses, title):
    """
    Plot the training and validation loss curves

    Args:
        train_losses: the list of training losses per epoch
        val_losses: the list of validation losses per epoch
        title: the title of the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def prepare_data(data_loader):
    """
    Prepare data by flattening and batching

    Args:
        data_loader: a dataLoader object

    Returns:
        the flattened input data and corresponding labels
    """
    inputs, labels = [], []
    for batch in data_loader:
        images, targets = batch
        images = images.view(images.size(0), -1)
        inputs.append(images)
        labels.append(targets)
    inputs = torch.cat(inputs)
    labels = torch.cat(labels)
    return inputs, labels


# normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: (x + 1) / 2)
])

mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_size = int(0.6 * len(mnist_data))
val_size = int(0.25 * len(mnist_data))
test_size = len(mnist_data) - train_size - val_size

train_data, val_data, test_data = random_split(mnist_data, [train_size, val_size, test_size])

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_inputs, train_labels = prepare_data(train_loader)
val_inputs, val_labels = prepare_data(val_loader)
test_inputs, test_labels = prepare_data(test_loader)

train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

input_dim = 784
hidden_layers = [128]
output_classes = 10

rbm_layers = []
current_input = train_inputs
current_val_input = val_inputs
for h_units in hidden_layers:
    rbm = RBM(input_dim, h_units).to(device)
    rbm_train_loss, rbm_val_loss = rbm.pretrain(current_input, current_val_input, rbm_batch_size=batch_size)
    current_input = rbm.extract_features(current_input)
    current_val_input = rbm.extract_features(current_val_input)
    rbm_layers.append(rbm)
    input_dim = h_units

    plot_losses(rbm_train_loss, rbm_val_loss, title="RBM training and validation losses")

dbn = DBN(rbm_layers=rbm_layers, output_classes=output_classes)
dbn.to(device)

dbn_train_loss, dbn_val_loss = train_dbn(dbn, train_inputs, train_labels, val_inputs, val_labels)

plot_losses(dbn_train_loss, dbn_val_loss, title="DBN Training and Validation Loss")

evaluate(dbn, test_inputs, test_labels)
