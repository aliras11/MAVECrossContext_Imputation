import torch
import torch.nn as nn
from torch.utils.data import Dataset

class DualColumnDataset(Dataset):
    def __init__(self, tensor1, mask1, tensor2, mask2):
        """
        Args:
            tensor1: First tensor (source)
            mask1: Mask for first tensor
            tensor2: Second tensor (target)
            mask2: Mask for second tensor
        """
        self.data1 = tensor1.T  # Transpose to get columns as samples
        self.mask1 = mask1.T
        self.data2 = tensor2.T
        self.mask2 = mask2.T
        if not (self.data1.shape == self.mask1.shape == self.data2.shape == self.mask2.shape):
            raise ValueError("All input tensors must have the same size.")
        
    def __len__(self):
        return self.data1.shape[0]
    
    def __getitem__(self, idx):
        return (self.data1[idx], self.mask1[idx], 
                self.data2[idx], self.mask2[idx])
    
class ColumnDataset(torch.utils.data.Dataset):
    """
    Each column (position) is a sample; features are amino-acid rows.
    x, mask: [A, N] with mask=True for observed entries.
    """
    def __init__(self, x: torch.Tensor, mask: torch.Tensor):
        assert x.shape == mask.shape, "x and mask must have same shape [A, N]"
        self.x = x.T.contiguous()      # [N, A]
        self.m = mask.T.contiguous()   # [N, A]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.m[idx]

class ColumnPairDataset(torch.utils.data.Dataset):
    """
    Each column (position) is a sample.
    x_src, y_tgt: [A, N]; mask_tgt: [A, N] (True for observed in target)
    used to map one map to another (e.g. wt -> av) directly. 
    """
    def __init__(self, x_src: torch.Tensor, y_tgt: torch.Tensor, mask_tgt: torch.Tensor):
        assert x_src.shape == y_tgt.shape == mask_tgt.shape, "all shapes must match [A, N]"
        self.x = x_src.T.contiguous()     # [N, A]
        self.y = y_tgt.T.contiguous()     # [N, A]
        self.m = mask_tgt.T.contiguous()  # [N, A] bool

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.m[idx]

class DualColumnAutoencoder(nn.Module):
    """
    Minimal bottleneck dual-head autoencoder: input -> bottleneck -> two output heads.
    Encoder: Linear(input_dim, encoding_dim) + Tanh
    Two decoders: Linear(encoding_dim, input_dim) each (reconstruction + prediction).
    """
    def __init__(self, input_dim, encoding_dim=12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim), nn.Tanh(),
        )
        self.decoder_recon = nn.Linear(encoding_dim, input_dim)
        self.decoder_pred = nn.Linear(encoding_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder_recon(encoded), self.decoder_pred(encoded)


class SimpleAutoencoder(nn.Module):
    """
    Minimal bottleneck autoencoder: input -> bottleneck -> output
    Single hidden layer with Tanh activation.
    """
    def __init__(self, input_dim: int, latent_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim), nn.Tanh(),
            nn.Linear(latent_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)

