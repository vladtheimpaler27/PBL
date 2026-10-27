import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from recommender import df, feature_matrix

X = torch.tensor(feature_matrix, dtype=torch.float32)
y = torch.tensor(LabelEncoder().fit_transform(df["playlist_genre"]), dtype=torch.long)
n_classes = int(y.max()) + 1

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

class GenreNet(nn.Module):
    def __init__(self, in_dim, n_classes, emb_dim=32):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, emb_dim), nn.ReLU(),
        )
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        return self.head(self.embed(x))

net = GenreNet(X.shape[1], n_classes)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(40):
    net.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss_fn(net(xb), yb).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(X_val).argmax(1) == y_val).float().mean().item()
    print(f"Epoch {epoch+1}/40  validation accuracy: {acc:.3f}")

with torch.no_grad():
    np.save("embeddings.npy", net.embed(X).numpy())
print("Saved embeddings.npy")