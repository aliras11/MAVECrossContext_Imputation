import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import (ColumnPairDataset, DualColumnDataset,
                    DualColumnAutoencoder, SimpleAutoencoder)

def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    se = (pred - target) ** 2
    se = se * mask.float()
    denom = mask.sum().clamp_min(eps)
    return se.sum() / denom


def _masked_mse_cross(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    se = (pred - target) ** 2
    se = se * mask.float()
    denom = mask.sum().clamp_min(eps)
    return se.sum() / denom


def zscore_by_feature(x, mask):
    # x: [F, N], mask: [F, N] with True for observed
    obs = mask.bool()
    F, N = x.shape
    means = torch.zeros(F, dtype=x.dtype)
    stds = torch.ones(F, dtype=x.dtype)
    for f in range(F):
        vals = x[f, obs[f]]
        if vals.numel() > 0:
            means[f] = vals.mean()
            s = vals.std(unbiased=False)
            stds[f] = torch.clamp(s, min=1e-6)
    xn = (x - means.unsqueeze(1)) / stds.unsqueeze(1)
    # Missing values are changed from global mean 
    return xn, means, stds

def un_zscore(xn, means, stds):
    # Supports per-feature (means.shape[0] == xn.shape[0]) and per-position (== xn.shape[1])
    if means.ndim != 1 or stds.ndim != 1:
        raise ValueError("means/stds must be 1D tensors.")
    if means.shape[0] == xn.shape[0] and stds.shape[0] == xn.shape[0]:
        # per-feature stats: broadcast across columns
        means_b = means.unsqueeze(1)
        stds_b = stds.unsqueeze(1)
    elif means.shape[0] == xn.shape[1] and stds.shape[0] == xn.shape[1]:
        # per-position stats: broadcast across rows
        means_b = means.unsqueeze(0)
        stds_b = stds.unsqueeze(0)
    else:
        raise ValueError("Shape mismatch: cannot broadcast means/stds to xn.")
    return xn * stds_b + means_b

def zscore_by_position(x, mask):
    # x: [F, N], mask: [F, N]
    obs = mask.bool()
    F, N = x.shape
    means = torch.zeros(N, dtype=x.dtype)
    stds  = torch.ones(N, dtype=x.dtype)
    for j in range(N):
        vals = x[obs[:, j], j]
        if vals.numel() > 0:
            means[j] = vals.mean()
            stds[j] = torch.clamp(vals.std(unbiased=False), min=1e-6)
    xn = (x - means.unsqueeze(0)) / stds.unsqueeze(0)
    return xn, means, stds

def impute_dual_columns(model, column1, column2,mask1, mask2):
    with torch.no_grad():
        model.eval()
        recon, pred = model(column1) #call model on the column from the base map 
        # Only replace missing values
        result_recon = torch.where(mask1.bool(), column1, recon)
        #there are two columns since we are imputing two maps
        result_pred = torch.where(mask2.bool(), column2, pred)
        return result_recon, result_pred


def impute_all_dual_columns(model, data_tensor1, data_tensor2, mask_tensor1, mask_tensor2):
    imputed_tensor1 = torch.zeros_like(data_tensor1)
    imputed_tensor2 = torch.zeros_like(data_tensor1)
    
    for i in range(data_tensor1.shape[1]):
        column1 = data_tensor1[:, i]
        column2 = data_tensor2[:, i]
        mask1 = mask_tensor1[:, i]
        mask2 = mask_tensor2[:, i]
        imputed_tensor1[:, i], imputed_tensor2[:, i] = impute_dual_columns(
            model, column1, column2, mask1, mask2)

    return imputed_tensor1, imputed_tensor2


def train_dual_column_imputer(src, src_train_mask,
                            tgt, tgt_train_mask,
                            encoder_dim=12,
                            epochs=400, batch_size=10, seed=None,
                            lr=0.001, weight_decay=0.0, recon_weight=0.3):
    """
    Train dual-output autoencoder for reconstruction and prediction

    Args:
        src: Source data tensor
        src_train_mask: Source mask tensor
        tgt: Target data tensor
        tgt_train_mask: Target mask tensor
        encoder_dim: Dimension of latent space (bottleneck)
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: L2 regularization
        recon_weight: Weight for reconstruction loss (prediction = 1 - recon_weight)
    """

    if seed is not None:
        torch.manual_seed(seed)

    input_dim = src.shape[0]
    model = DualColumnAutoencoder(input_dim, encoding_dim=encoder_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    src_n, s_mu, s_sd = zscore_by_position(src, src_train_mask)
    tgt_n, t_mu, t_sd = zscore_by_position(tgt, tgt_train_mask)

    dataset = DualColumnDataset(src_n, src_train_mask, tgt_n, tgt_train_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    pred_weight = 1.0 - recon_weight

    # Training loop
    for epoch in range(epochs):
        total_recon_loss = 0
        total_pred_loss = 0

        for batch_data1, batch_mask1, batch_data2, batch_mask2 in dataloader:
            optimizer.zero_grad()

            recon, pred = model(batch_data1)
            recon_loss = torch.sum((recon - batch_data1)**2 * batch_mask1) / (torch.sum(batch_mask1) + 1e-8)
            pred_loss = torch.sum((pred - batch_data2)**2 * batch_mask2) / (torch.sum(batch_mask2) + 1e-8)

            loss = recon_weight * recon_loss + pred_weight * pred_loss

            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_pred_loss += pred_loss.item()
    

    imputed_src_n, imputed_tgt_n = impute_all_dual_columns(model, src_n, tgt_n, src_train_mask, tgt_train_mask)
    imputed_src = un_zscore(imputed_src_n, s_mu, s_sd)
    imputed_tgt = un_zscore(imputed_tgt_n, t_mu, t_sd)
    return model, (s_mu, s_sd, t_mu, t_sd), imputed_src, imputed_tgt


def train_cross_map_with_simpleae(
    src: torch.Tensor,              # [F, N] source map (e.g., wt200)
    tgt: torch.Tensor,              # [F, N] target map (e.g., av12)
    src_train_mask: torch.Tensor,         # [F, N] bool (observed in src)
    tgt_train_mask: torch.Tensor,   # [F, N] bool (observed in tgt for training)
    latent_dim: int = 12,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int | None = None,
    verbose: bool = False,
    impute_mask: torch.Tensor | None = None  # where True keep tgt, False fill with prediction
):
    """
    Map src -> tgt with SimpleAutoencoder and masked MSE using (src_mask & tgt_train_mask).
    Normalizes both src/tgt per feature using the same joint mask.
    Returns:
      model,
      (s_mu, s_sd, t_mu, t_sd),
      tgt_pred_norm_filled (z-scale),
      tgt_pred_filled (orig scale)
    """
    if seed is not None:
        torch.manual_seed(seed)

    joint_mask = src_train_mask.bool() & tgt_train_mask.bool()
    # Normalize per-feature using joint observed entries
    src_n, s_mu, s_sd = zscore_by_position(src, joint_mask)
    tgt_n, t_mu, t_sd = zscore_by_position(tgt, joint_mask)
    ds = ColumnPairDataset(src_n, tgt_n, joint_mask)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    F = src.shape[0]
    model = SimpleAutoencoder(input_dim=F, latent_dim=latent_dim)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        steps = 0
        for x_b, y_b, mt_b in dl:
            opt.zero_grad()
            pred_b = model(x_b)  # predict target column on z-scale
            loss = _masked_mse(pred_b, y_b, mt_b)
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"[CrossSimpleAE] epoch {epoch:4d} loss={total/max(1,steps):.6f}")

    # Impute over all columns: keep observed target, fill the rest with prediction from src
    model.eval()
    with torch.no_grad():
        tgt_pred_n = torch.zeros_like(tgt_n)
        N = src.shape[1]
        for i in range(N):
            x_col = src_n[:, i].unsqueeze(0)      # [1, F]
            pred_col = model(x_col).squeeze(0)    # [F] normalized prediction
            keep_mask = impute_mask[:, i].bool() if impute_mask is not None else tgt_train_mask[:, i].bool()
            filled_col = torch.where(keep_mask, tgt_n[:, i], pred_col)
            tgt_pred_n[:, i] = filled_col

    tgt_pred = un_zscore(tgt_pred_n, t_mu, t_sd)
    return model, (s_mu, s_sd, t_mu, t_sd), tgt_pred_n, tgt_pred


