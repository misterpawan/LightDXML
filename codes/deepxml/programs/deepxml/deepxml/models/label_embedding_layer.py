import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

class DistanceInverseProportionalLoss(nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sum_ = torch.sum((input - target) ** 2, dim=1)
        ret = torch.mean(torch.log(sum_ + torch.ones(sum_.size(), device=sum_.device)))
        return ret


class LabelEmbed(nn.Module):

    def __init__(self, vocab_dims, embed_dims, hidden_dims, device="cuda:0"):
        super(LabelEmbed, self).__init__()

        self.device = torch.device(device)
        self.embedding_layer = nn.Embedding(vocab_dims, embed_dims)
        self.fc_layer = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.Linear(hidden_dims, embed_dims)
        )

        for params in self.embedding_layer.parameters():
            params.requires_grad = False

    def initialize(self, doc_embeddings, dataset):
        print("Initializing label embeddings")
        self.embedding_layer.weight = nn.Parameter(torch.zeros(self.embedding_layer.weight.size()).to(self.device))
        counts = [0] * self.embedding_layer.weight.size(0)
        for idx, indices in tqdm(enumerate(dataset), total=len(dataset)):
            for ind in indices:
                counts[ind] += 1
                self.embedding_layer.weight[ind] += doc_embeddings[idx]
        for i in range(0, self.embedding_layer.weight.size(0)):
            if counts[i] > 0:
                self.embedding_layer.weight[i] /= counts[i]
        # norms = torch.linalg.norm(self.embedding_layer.weight, dim=1)
        # for i in range(0, self.embedding_layer.weight.size(0)):
        #     if norms[i] > 0:
        #         self.embedding_layer.weight[i] /= norms[i]


    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.fc_layer(x)
        return x

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def model_size(self):  # Assumptions: 32bit floats
        return self.num_params * 4 / math.pow(2, 20)
