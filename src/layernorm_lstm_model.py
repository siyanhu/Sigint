import torch

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IMULSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super(IMULSTMModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.device = get_device()
        
        # Initial normalization layer
        self.initial_norm = torch.nn.LayerNorm(input_size)
        
        # Create a list of LSTM layers
        self.lstm_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(torch.nn.LSTM(input_size, hidden_sizes[0], batch_first=True))
        self.norm_layers.append(torch.nn.LayerNorm(hidden_sizes[0]))
        self.dropout_layers.append(torch.nn.Dropout(dropout_rate))
        
        # Subsequent LSTM layers
        for i in range(1, self.num_layers):
            self.lstm_layers.append(torch.nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
            self.norm_layers.append(torch.nn.LayerNorm(hidden_sizes[i]))
            self.dropout_layers.append(torch.nn.Dropout(dropout_rate))
        
        # Final fully connected layer
        self.fc = torch.nn.Linear(hidden_sizes[-1], output_size)
        
        # Move the model to the appropriate device
        self.to(self.device)
        
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Apply initial normalization
        x = self.initial_norm(x)
        
        # Initialize hidden and cell states
        h = [torch.zeros(1, x.size(0), hidden_size, device=self.device) for hidden_size in self.hidden_sizes]
        c = [torch.zeros(1, x.size(0), hidden_size, device=self.device) for hidden_size in self.hidden_sizes]
        
        # Pass through LSTM layers
        for i, (lstm, norm, dropout) in enumerate(zip(self.lstm_layers, self.norm_layers, self.dropout_layers)):
            x, (h[i], c[i]) = lstm(x, (h[i], c[i]))
            x = norm(x)
            x = dropout(x)
        
        # Pass through the fully connected layer
        out = self.fc(x[:, -1, :])
        return out

    def to(self, device):
        self.device = device
        return super(IMULSTMModel, self).to(device)