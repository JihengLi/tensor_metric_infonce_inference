import torch
from torch.nn.functional import interpolate


def znorm_nonzero(t: torch.Tensor) -> torch.Tensor:
    mask = t != 0
    if mask.any():
        vals = t[mask]
        mu, std = vals.mean(), vals.std()
        if std > 0:
            t = torch.where(mask, (t - mu) / std, t)
    return t


def resize_to_64(t: torch.Tensor) -> torch.Tensor:
    out = interpolate(
        t.unsqueeze(0),
        size=(64, 64, 64),
        mode="trilinear",
        align_corners=False,
    ).squeeze(
        0
    )  # (6, 64, 64, 64)
    return out


def clean_tensor(t: torch.Tensor) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=0.0)
    finite = t[torch.isfinite(t)]
    if finite.numel() > 0:
        mx, mn = finite.max(), finite.min()
        t = torch.where(t == float("inf"), mx, t)
        t = torch.where(t == float("-inf"), mn, t)
    return t
