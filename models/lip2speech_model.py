import torch
import torch.nn as nn

VOCAB_SIZE = 100

class LipToSpeech(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden: int = 256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),       
            nn.ReLU(),
            nn.MaxPool2d(2),      

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Dropout(0.3)
        )

        self.lstm = nn.LSTM(
            input_size=64 * 24 * 24,    
            hidden_size=hidden,
            num_layers=1,              
            batch_first=True,
            dropout=0.0,
            bidirectional=False     
        )

        lstm_out_dim = hidden
        self.fc = nn.Linear(lstm_out_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = self.cnn(x)             
        x = x.reshape(B, T, -1)       
        x, _ = self.lstm(x)          
        x = self.fc(x)               

        return x                       