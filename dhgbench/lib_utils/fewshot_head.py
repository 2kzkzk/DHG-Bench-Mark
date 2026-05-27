import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoHead(nn.Module):
    """Episode-local prototype classifier for node embeddings."""

    def __init__(self, metric="cosine", temperature=10.0):
        super().__init__()
        if metric not in ["cosine", "euclidean"]:
            raise ValueError(f"Unsupported prototype metric: {metric}")
        self.metric = metric
        self.temperature = temperature

    def forward(self, z, support_idx, support_y, query_idx, way):
        support_z = z[support_idx]
        query_z = z[query_idx]

        prototypes = []
        for class_id in range(way):
            class_mask = support_y == class_id
            if not torch.any(class_mask):
                raise ValueError(f"Episode support set has no samples for local class {class_id}.")
            prototypes.append(support_z[class_mask].mean(dim=0))
        prototypes = torch.stack(prototypes, dim=0)

        if self.metric == "cosine":
            query_z = F.normalize(query_z, p=2, dim=-1)
            prototypes = F.normalize(prototypes, p=2, dim=-1)
            logits = self.temperature * (query_z @ prototypes.t())
        else:
            logits = -torch.cdist(query_z, prototypes).pow(2)

        return logits
