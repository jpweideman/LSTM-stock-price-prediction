# import torch
# import torch.nn as nn


# class LSTMStockPricePredictor(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
#         super(LSTMStockPricePredictor, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.lstm = nn.LSTM(input_dim, hidden_dim,
#                             num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0),
#                          self.hidden_dim).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0),
#                          self.hidden_dim).to(x.device)

#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out


# if __name__ == "__main__":
#     model = LSTMStockPricePredictor(5, 64, 2, 1)
#     print(f"Model architecture: {model}")
#     print(
#         f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

#     dummy_input = torch.randn(1, 5, 5)
#     output = model(dummy_input)
#     print(f"Output shape: {output.shape}")






import torch
import torch.nn as nn

class LSTMStockPricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMStockPricePredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # * 2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # * 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # * 2 for bidirectional
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    model = LSTMStockPricePredictor(5, 32, 2, 1, 0)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 5, 5)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")