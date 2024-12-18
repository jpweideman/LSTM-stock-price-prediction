import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


def load_data(seq_length=50):
    file_path = 'data_relevant_SMSN.csv'
    data = pd.read_csv(file_path)
    data.dropna(axis=1, inplace=True)

    # Select relevant features and the target
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # drop the VWAP column and make the the target
    target = data['VWAP']
    columns_to_remove = ['VWAP']
    features = data.drop(columns=columns_to_remove)
    

    print(target.head())
    # Normalize the data
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Normalize the target values
    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

    # Convert to PyTorch tensors
    X = torch.tensor(features_scaled, dtype=torch.float32)
    y = torch.tensor(target_scaled, dtype=torch.float32)

    # Prepare the data for LSTM
    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return torch.stack(X_seq), torch.stack(y_seq)

    X_seq, y_seq = create_sequences(X, y, seq_length)

    # Split the data into training and testing sets while preserving the temporal order
    split_index = int(0.95 * len(X_seq))
    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train, y_test = y_seq[:split_index], y_seq[split_index:]

    return features_scaled, target_scaler,  X_train, X_test, y_train, y_test
