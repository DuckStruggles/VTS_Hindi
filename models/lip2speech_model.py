import torch
import torch.nn as nn

VOCAB_SIZE = 32   # matches kmeans_hindi_units.py n_clusters=100

class LipToSpeech(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden: int = 256):
        super().__init__()

        # --- spatial feature extractor ---
        # Input frames are 96×96. After two MaxPool2d(2) the spatial size
        # becomes 24×24, so the flattened size is 64 * 24 * 24 = 36864.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),          # added: stabilises training
            nn.ReLU(),
            nn.MaxPool2d(2),             # 96 → 48

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),             # 48 → 24
            nn.Dropout(0.3)
        )

        # --- temporal sequence model ---
        self.lstm = nn.LSTM(
            input_size=64 * 24 * 24,     # 36864
            hidden_size=hidden,
            num_layers=1,                # added: extra depth helps Hindi
            batch_first=True,
            dropout=0.0,
            bidirectional=False           # added: sees future context too
        )

        # bidirectional doubles the output dimension
        lstm_out_dim = hidden

        # --- unit classifier: outputs one logit per K-means cluster ---
        self.fc = nn.Linear(lstm_out_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # apply CNN to every frame independently
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)                  # (B*T, 64, 24, 24)

        # flatten spatial dims
        x = x.reshape(B, T, -1)          # (B, T, 36864)

        # temporal modelling
        x, _ = self.lstm(x)              # (B, T, hidden*2)

        # per-frame logits over vocabulary
        x = self.fc(x)                   # (B, T, VOCAB_SIZE)

        return x                         # raw logits – no softmax here
                                         # use CrossEntropyLoss during training
                                         # use argmax at inference