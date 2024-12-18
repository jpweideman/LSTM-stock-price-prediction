import torch
from lstm_model import LSTMStockPricePredictor
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from lstm_make_dataset import load_data
from matplotlib import pyplot as plt


# Load data
features_scaled, target_scaler, X_train, X_test, y_train, y_test = load_data()
test_data = y_test
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    print("Predicting")
    model.load_state_dict(torch.load("lstm_model.pth"))
    model.eval()
    y_pred = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            y_pred.extend(outputs.numpy())

    y_pred = target_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
    y_test_scaled = target_scaler.inverse_transform(
        y_test.numpy().reshape(-1, 1))

    results = pd.DataFrame(
        {'Actual': y_test_scaled.flatten(), 'Predicted': y_pred.flatten()})

    # Save results to CSV if needed
    plt.figure(figsize=(10, 6))
    plt.plot(results['Actual'], label='Actual Values')
    plt.plot(results['Predicted'], label='Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Stock Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend()
    plt.grid(True)
    plt.savefig('lstm_predictions.png')

    return results


if __name__ == "__main__":
    input_dim = features_scaled.shape[1]
    hidden_dim = 32
    num_layers = 2
    output_dim = 1

    model = LSTMStockPricePredictor(
        input_dim, hidden_dim, num_layers, output_dim)
    predictions = predict(model=model, dataloader=test_loader)
    print("DONE")
