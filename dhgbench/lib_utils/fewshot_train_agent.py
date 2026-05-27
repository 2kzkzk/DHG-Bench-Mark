import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lib_utils.fewshot_head import ProtoHead


def train_fewshot_node_cls(model, data, episode_banks, args):
    """Train an encoder with an episode-local prototype head."""
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    proto_head = ProtoHead(metric=args.fs_metric, temperature=args.fs_temperature).to(args.device)
    trainable_param_count = sum(param.numel() for param in trainable_params)
    total_param_count = sum(param.numel() for param in model.parameters())

    if not trainable_params:
        print("Few-shot model has no trainable parameters; skipping episodic optimization.")
        start_time = time.time()
        val_acc, val_std, val_ci95 = evaluate_fewshot_node_cls(
            model,
            data,
            episode_banks["val"],
            args,
        )
        train_time = time.time() - start_time
        print(f"Validation episodic acc: {val_acc:.4f} +- {val_std:.4f} (95% CI {val_ci95:.4f})")
        return model, {
            "best_val_acc": val_acc,
            "best_val_std": val_std,
            "best_val_ci95": val_ci95,
            "best_episode": 0,
            "training_time": train_time,
            "trainable_param_count": trainable_param_count,
            "total_param_count": total_param_count,
            "notes": "no trainable params",
        }

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.wd)
    best_val_acc = -1.0
    best_val_std = 0.0
    best_val_ci95 = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    last_improved_episode = 0
    start_time = time.time()
    eval_interval = max(1, int(args.fs_eval_interval))

    for episode_id, batch in enumerate(tqdm(episode_banks["train"]), start=1):
        model.train()
        optimizer.zero_grad()

        z = _node_embeddings(model, data)
        _check_embedding_shape(z, args)
        way = _episode_way(batch)
        logits = proto_head(
            z,
            batch["support_idx"].to(args.device),
            batch["support_y"].to(args.device),
            batch["query_idx"].to(args.device),
            way,
        )
        expected_shape = (way * args.fs_query, way)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(f"Train episode logits shape {tuple(logits.shape)} != expected {expected_shape}.")

        loss = F.cross_entropy(logits, batch["query_y"].to(args.device))
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_thresh)
        optimizer.step()

        should_eval = (episode_id % eval_interval == 0) or (episode_id == len(episode_banks["train"]))
        if should_eval:
            val_acc, val_std, val_ci95 = evaluate_fewshot_node_cls(
                model,
                data,
                episode_banks["val"],
                args,
            )
            print(
                f"Episode: {episode_id:04d}, Training loss: {loss.item():.4f}, "
                f"Val Acc: {val_acc:.4f} +- {val_std:.4f} (95% CI {val_ci95:.4f})"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_std = val_std
                best_val_ci95 = val_ci95
                best_model_state = copy.deepcopy(model.state_dict())
                last_improved_episode = episode_id
            elif (
                args.early_stop
                and args.fs_patience_episodes > 0
                and episode_id - last_improved_episode >= args.fs_patience_episodes
            ):
                print(f"Early stopping at episode {episode_id}; best val acc={best_val_acc:.4f}.")
                break

    model.load_state_dict(best_model_state)
    train_time = time.time() - start_time
    print(f"Few-shot training time: {train_time:.2f}")
    return model, {
        "best_val_acc": best_val_acc,
        "best_val_std": best_val_std,
        "best_val_ci95": best_val_ci95,
        "best_episode": last_improved_episode,
        "training_time": train_time,
        "trainable_param_count": trainable_param_count,
        "total_param_count": total_param_count,
        "notes": "trainable encoder",
    }


@torch.no_grad()
def evaluate_fewshot_node_cls(model, data, episodes, args):
    """Evaluate average episodic accuracy over a fixed episode bank."""
    if len(episodes) <= 0:
        raise ValueError("Episode bank must contain at least one episode.")

    proto_head = ProtoHead(metric=args.fs_metric, temperature=args.fs_temperature).to(args.device)
    model.eval()
    z = _node_embeddings(model, data)
    _check_embedding_shape(z, args)

    accs = []
    for batch in episodes:
        way = _episode_way(batch)
        logits = proto_head(
            z,
            batch["support_idx"].to(args.device),
            batch["support_y"].to(args.device),
            batch["query_idx"].to(args.device),
            way,
        )
        expected_shape = (way * args.fs_query, way)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(f"Evaluation episode logits shape {tuple(logits.shape)} != expected {expected_shape}.")

        pred = logits.argmax(dim=1)
        acc = (pred == batch["query_y"].to(args.device)).float().mean().item()
        accs.append(acc)

    accs = np.asarray(accs, dtype=np.float64)
    mean_acc = float(accs.mean())
    std_acc = float(accs.std())
    ci95 = float(1.96 * std_acc / np.sqrt(len(accs)))
    return mean_acc, std_acc, ci95


@torch.no_grad()
def infer_fewshot_embedding_dim(model, data, args):
    model.eval()
    z = _node_embeddings(model, data)
    _check_embedding_shape(z, args)
    return int(z.size(1))


def _node_embeddings(model, data):
    out = model(data)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _check_embedding_shape(z, args):
    if z.dim() != 2:
        raise RuntimeError(f"fewshot_node_cls expects a 2-D node embedding tensor, got shape {tuple(z.shape)}.")
    if _allows_arbitrary_embedding_dim(args):
        return
    if z.size(1) != args.embedding_hidden:
        raise RuntimeError(
            f"fewshot_node_cls expects embedding dimension args.embedding_hidden={args.embedding_hidden}, "
            f"but model returned {z.size(1)}."
        )


def _episode_way(batch):
    return len(batch["classes"])


def _allows_arbitrary_embedding_dim(args):
    if args.method == "RawFeatureProto":
        return True
    return args.method == "ZEN" and args.zen_mode in ["no_projection", "raw_feature_proto"]
