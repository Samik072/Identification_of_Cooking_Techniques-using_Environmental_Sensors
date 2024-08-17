import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

class DeepModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepModel, self).__init__()
        layers = []
        in_features = input_size

        # Input normalization
        layers.append(nn.BatchNorm1d(input_size))

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            # Batch normalization for hidden layers
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def ann_model(X, y, test_size = 0.2):
    # Evaluate the best model on the test set
    def evaluate_accuracy(model, data_loader):
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        accuracy = total_correct / total_samples
        return accuracy

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Convert data to PyTorch dataset
    dataset = data.TensorDataset(X, y)

    # Split the data into train, dev, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create PyTorch DataLoader for each set
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    dev_loader = data.DataLoader(data.TensorDataset(X_dev, y_dev), batch_size=32, shuffle=False)
    test_loader = data.DataLoader(data.TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    # Define hyperparameters to try
    learning_rate = 0.01
    num_epochs = 50

    # Function to train and evaluate the model
    def train_and_evaluate(model, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Lists to store convergence data
        train_losses = []
        dev_losses = []
        train_accuracies = []
        dev_accuracies = []

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Compute and store losses and accuracies
            train_loss = loss.item()
            train_losses.append(train_loss)
            train_acc = evaluate_accuracy(model, train_loader)
            train_accuracies.append(train_acc)
            dev_acc = evaluate_accuracy(model, dev_loader)
            dev_accuracies.append(dev_acc)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Dev Accuracy: {dev_acc:.4f}")

        # Plot convergence graph
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
        plt.plot(range(1, num_epochs + 1), dev_accuracies, label='Dev Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Evaluation
        model.eval()
        total_correct = 0
        total_samples = 0
        predicted_labels = []
        with torch.no_grad():
            for inputs, labels in dev_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                predicted_labels.extend(predicted.tolist())

        accuracy = total_correct / total_samples

        # Classification report and F1-score
        print("Classification Report:")
        print(classification_report(y_dev, predicted_labels))
        f1 = f1_score(y_dev, predicted_labels, average='weighted')
        print(f"F1-Score: {f1:.4f}")

        return accuracy

    # Define different configurations for hidden layers and units
    hidden_layers_configs = [
        [16, 8],  # 2 hidden layers with 16 and 8 units
        [32, 16, 8],  # 3 hidden layers with 32, 16, and 8 units
        [64, 32, 16, 8]  # 4 hidden layers with 64, 32, 16, and 8 units
    ]

    # Train and evaluate models with different configurations
    best_accuracy = 0.0
    best_model = None
    for hidden_layers in hidden_layers_configs:
        model = DeepModel(input_size=X.shape[1], hidden_sizes=hidden_layers, output_size=y.shape[0])
        accuracy = train_and_evaluate(model, learning_rate)
        print(f"Hidden layers configuration: {hidden_layers}, Accuracy: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f"Best model hidden layers configuration: {best_model.layers}, Best accuracy: {best_accuracy}")
    test_accuracy = evaluate_accuracy(best_model, test_loader)
    print(f"Test Accuracy: {test_accuracy}")
    return best_model
