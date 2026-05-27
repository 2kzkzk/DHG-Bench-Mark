import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class ZENEncoder(nn.Module):
    """A ZEN-style hypergraph encoder adapter that returns node embeddings."""

    def __init__(self, num_features, num_targets, args):
        super().__init__()
        self.mode = args.zen_mode
        coeffs = _parse_coefficients(args.zen_hyperparams)
        self.register_buffer("coefficients", torch.tensor(coeffs, dtype=torch.float))

        if self.mode == "trainable_adapter":
            self.adapter = nn.Linear(num_features, num_targets, bias=False)
            self.register_buffer("projection", torch.empty(0))
        elif self.mode == "random_projection" and num_features != num_targets:
            self.adapter = None
            generator = torch.Generator()
            generator.manual_seed(int(args.zen_projection_seed))
            projection = torch.randn(num_features, num_targets, generator=generator)
            projection = projection / math.sqrt(max(1, num_targets))
            self.register_buffer("projection", projection)
        else:
            self.adapter = None
            self.register_buffer("projection", torch.empty(0))

    def reset_parameters(self):
        if self.adapter is not None:
            self.adapter.reset_parameters()

    def forward(self, data):
        if self.mode == "raw_feature_proto":
            z = data.x
        else:
            H = data.hyperedge_index
            X = data.x
            z = _zen_embedding(H, X, self.coefficients.to(X.device, X.dtype))
        if self.adapter is not None:
            z = self.adapter(z)
        if self.projection.numel() > 0:
            z = z @ self.projection.to(X.device, X.dtype)
        z = F.normalize(z, p=2, dim=-1)
        return z, None


class RawFeatureProto(nn.Module):
    """Non-parametric baseline that classifies raw node features with ProtoHead."""

    def __init__(self, num_features, num_targets, args):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, data):
        return data.x, None


def _parse_coefficients(raw_value):
    parts = [part.strip() for part in str(raw_value).replace("/", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--zen_hyperparams must contain three comma-separated coefficients.")
    coeffs = [float(part) for part in parts]
    total = sum(coeffs)
    if total <= 0:
        raise ValueError("--zen_hyperparams coefficients must sum to a positive value.")
    return [value / total for value in coeffs]


def _degree_norms(H, dtype):
    device = H.device
    ones = torch.ones(H.shape[1], dtype=dtype, device=device)
    dv = scatter(src=ones, index=H[0], dim=0, reduce="sum")
    dv_1 = dv - 1
    de_1 = scatter(src=ones, index=H[1], dim=0, reduce="sum") - 1

    dv = dv.pow(-0.5)
    dv_1 = dv_1.pow(-1.0)
    de_1 = de_1.pow(-1.0)

    dv[dv.isinf()] = 0
    dv_1[dv_1.isinf()] = 0
    de_1[de_1.isinf()] = 0

    return dv.unsqueeze(1), dv_1.unsqueeze(1), de_1.unsqueeze(1)


def _rsis(H, dv_1, de_1):
    rsi_1 = scatter(src=de_1[H[1]], index=H[0], dim=0, reduce="sum")

    de2 = de_1.pow(2.0)
    rsi_2 = scatter(src=dv_1[H[0]], index=H[1], dim=0, reduce="sum")
    rsi_2 = de2 * rsi_2
    rsi_2 = scatter(src=rsi_2[H[1]], index=H[0], dim=0, reduce="sum")

    correction_term = scatter(src=de2[H[1]], index=H[0], dim=0, reduce="sum")
    rsi_2 = rsi_2 - dv_1 * correction_term

    return rsi_1, rsi_2


def _propagation(H, X, de_1, rsi_1):
    z = scatter(src=X[H[0]], index=H[1], dim=0, reduce="sum")
    z = de_1 * z
    z = scatter(src=z[H[1]], index=H[0], dim=0, reduce="sum")
    return z - rsi_1 * X


def _zen_embedding(H, X, coefficients):
    dv, dv_1, de_1 = _degree_norms(H, X.dtype)
    rsi_1, rsi_2 = _rsis(H, dv_1, de_1)

    z1 = dv * X
    ax = _propagation(H, z1, de_1, rsi_1)

    z2 = dv_1 * ax
    aax = _propagation(H, z2, de_1, rsi_1)
    aax = aax - rsi_2 * z1

    ax = dv * ax
    aax = dv * aax

    z = coefficients[0] * X + coefficients[1] * ax + coefficients[2] * aax
    return F.normalize(z, p=2, dim=-1)
