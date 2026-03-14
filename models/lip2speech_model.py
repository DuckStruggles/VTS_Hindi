import torch
import torch.nn as nn

class LipToSpeech(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.lstm = nn.LSTM(
            input_size=64*24*24,
            hidden_size=256,
            batch_first=True
        )

        self.fc = nn.Linear(256,100)

    def forward(self,x):

        B,T,C,H,W = x.shape

        x = x.view(B*T,C,H,W)

        x = self.cnn(x)

        x = x.reshape(B,T,-1)

        x,_ = self.lstm(x)

        x = self.fc(x)

        return x