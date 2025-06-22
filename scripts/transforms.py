import torch
from torch.nn.functional import interpolate


def _tensor_to_eigenvalues(t: torch.Tensor) -> torch.Tensor:
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = t
    A = torch.stack(
        [
            torch.stack([Dxx, Dxy, Dxz], 0),
            torch.stack([Dxy, Dyy, Dyz], 0),
            torch.stack([Dxz, Dyz, Dzz], 0),
        ]
    )  # shape (3, 3, D, H, W)
    A = A.permute(2, 3, 4, 0, 1).reshape(-1, 3, 3)  # (N, 3, 3)
    w = torch.linalg.eigvalsh(A).real  # (N, 3)
    w = w.flip(-1)
    w = w.reshape(t.shape[1], t.shape[2], t.shape[3], 3).permute(3, 0, 1, 2)
    return w.float()


def _jitter_background(t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    mask = t == 0
    noise = torch.randn_like(t) * std
    return torch.where(mask, noise, t)


def _znorm_nonzero(t: torch.Tensor) -> torch.Tensor:
    mask = t != 0
    if mask.any():
        vals = t[mask]
        mu, std = vals.mean(), vals.std()
        if std > 0:
            t = torch.where(mask, (t - mu) / std, t)
    return t


def _resize_to_64(t: torch.Tensor) -> torch.Tensor:
    out = interpolate(
        t.unsqueeze(0),
        size=(64, 64, 64),
        mode="trilinear",
        align_corners=False,
    ).squeeze(
        0
    )  # (C, 64, 64, 64)
    return out


def _clean_tensor(t: torch.Tensor) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=0.0)
    finite = t[torch.isfinite(t)]
    if finite.numel() > 0:
        mx, mn = finite.max(), finite.min()
        t = torch.where(t == float("inf"), mx, t)
        t = torch.where(t == float("-inf"), mn, t)
    return t
