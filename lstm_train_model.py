# import os
# import click
# import matplotlib.pyplot as plt
# import torch
# from lstm_model import LSTMStockPricePredictor
# import pandas as pd
# # from data import corrupt_mnis
# from lstm_make_dataset import load_data
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset
# import torch.nn as nn
# import torch.optim as optim


# DEVICE = torch.device("cuda" if torch.cuda.is_available(
# ) else "mps" if torch.backends.mps.is_available() else "cpu")


# # print("Current directory:", os.getcwd())

# @click.group()
# def cli():
#     """Command line interface."""
#     pass


# @click.command()
# @click.option("--lr", default=1e-3, help="learning rate to use for training")
# @click.option("--batch_size", default=32, help="batch size to use for training")
# @click.option("--epochs", default=10, help="number of epochs to train for")
# def train(lr, batch_size, epochs) -> None:
#     print("Training day and night")
#     print(f"{lr=}, {batch_size=}, {epochs=}")

#     # Train loader
#     features_scaled, _, X_train, X_test, y_train, y_test = load_data()

#     train_dataset = TensorDataset(X_train, y_train)
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

#     # Initialize the model, loss function, and optimizer
#     input_dim = features_scaled.shape[1]
#     hidden_dim = 64
#     num_layers = 2
#     output_dim = 1
#     model = LSTMStockPricePredictor(
#         input_dim, hidden_dim, num_layers, output_dim)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     statistics = {"train_loss": [], "train_accuracy": []}
#     # Training with loss tracking
#     losses = []
#     num_epochs = 50
#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0
#         for X_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             if torch.isnan(loss):
#                 raise ValueError(f"Loss became NaN at epoch {epoch}")
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#         avg_epoch_loss = epoch_loss / len(train_loader)
#         losses.append(avg_epoch_loss)
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

#     print("Training complete")
#     torch.save(model.state_dict(), "lstm_model.pth")
#     print("Model saved")
#     # plot the loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(losses, label='Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title("LSTM Training Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("lstm_training_loss.png")
#     print("Loss plot saved")


# if __name__ == "__main__":
#     train()










import os
import click
import matplotlib.pyplot as plt
import torch
from lstm_model import LSTMStockPricePredictor
import pandas as pd
from lstm_make_dataset import load_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=32, help="batch size to use for training")
@click.option("--epochs", default=30, help="number of epochs to train for")
def train(lr, batch_size, epochs) -> None:
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # Train loader
    features_scaled, _, X_train, X_test, y_train, y_test = load_data()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_dim = features_scaled.shape[1]
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    model = LSTMStockPricePredictor(input_dim, 
                                    hidden_dim, num_layers, output_dim, dropout=0
                                    ).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    statistics = {"train_loss": [], "train_accuracy": []}
    # Training with loss tracking
    losses = []
    num_epochs = epochs
    best_loss = float('inf')
    patience, trials = 10, 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            if torch.isnan(loss):
                raise ValueError(f"Loss became NaN at epoch {epoch}")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

        scheduler.step(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            trials = 0
            torch.save(model.state_dict(), "lstm_model.pth")
        else:
            trials += 1
            if trials >= patience:
                print(f"Early stopping on epoch {epoch+1}")
                break

    print("Training complete")
    print("Model saved")
    # plot the loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("LSTM Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("lstm_training_loss.png")
    print("Loss plot saved")

if __name__ == "__main__":
    train()