import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class Autoencoder(nn.Module):

    def __init__(self, vocab_dims, embed_dims):
        super(Autoencoder, self).__init__()

        self.vocab_dims = vocab_dims
        self.encoder = nn.Sequential(
            nn.Linear(vocab_dims, 2048),
            nn.Linear(2048, embed_dims)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dims, 2048),
            nn.Linear(2048, vocab_dims),
            nn.Softmax(dim=1)
        )

    def encode(self, x):
        x = F.one_hot(x, self.vocab_dims).to(torch.float32)
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = F.one_hot(x, self.vocab_dims).to(torch.float32)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(model, X, epochs, batch_size):
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    X = torch.LongTensor(X)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0
        for i in tqdm(range(0, X.size(0), batch_size), desc=f"Training for epoch {epoch+1}"):
            words_from = X[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(words_from), words_from)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * words_from.size(0)
        scheduler.step()

        print(f"Training for epoch {epoch+1} done. Cross Entropy loss = {running_loss / X.shape[0]}")

