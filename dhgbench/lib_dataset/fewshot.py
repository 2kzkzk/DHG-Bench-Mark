import hashlib
import json
import os
import re
import warnings

import torch


def parse_class_split(split_str):
    """Parse an explicit class split string in train/val/test order."""
    if not isinstance(split_str, str) or re.fullmatch(r"\d+/\d+/\d+", split_str.strip()) is None:
        raise ValueError(
            f"Invalid --fs_class_split={split_str!r}. Expected format is 'int/int/int', e.g. '3/2/2'."
        )

    train_num, val_num, test_num = [int(part) for part in split_str.strip().split("/")]
    if train_num <= 0 or val_num <= 0 or test_num <= 0:
        raise ValueError(
            f"Invalid --fs_class_split={split_str!r}. Train/val/test class counts must all be greater than 0."
        )

    return train_num, val_num, test_num


def split_classes(
    y,
    seed,
    fs_class_split=None,
    val_ratio=0.2,
    test_ratio=0.2,
    fixed_order=False,
):
    """Create class-disjoint train/val/test splits for episodic node classification."""
    labels = y.detach().cpu().view(-1)
    classes = [int(cls) for cls in torch.unique(labels, sorted=True).tolist()]
    num_classes = len(classes)

    if num_classes == 0:
        raise ValueError("Cannot split classes because no labels were found.")

    if fixed_order:
        ordered_classes = classes
    else:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        perm = torch.randperm(num_classes, generator=generator).tolist()
        ordered_classes = [classes[idx] for idx in perm]

    if fs_class_split is not None:
        warnings.warn(
            "--fs_class_split was provided; --fs_val_class_ratio and --fs_test_class_ratio are ignored.",
            stacklevel=2,
        )
        train_num, val_num, test_num = parse_class_split(fs_class_split)
        requested = train_num + val_num + test_num
        if requested > num_classes:
            raise ValueError(
                f"--fs_class_split={fs_class_split} requests {requested} classes, "
                f"but dataset has only {num_classes} classes."
            )
    else:
        if val_ratio < 0 or test_ratio < 0 or val_ratio >= 1 or test_ratio >= 1:
            raise ValueError("--fs_val_class_ratio and --fs_test_class_ratio must be in [0, 1).")
        val_num = max(1, int(num_classes * val_ratio))
        test_num = max(1, int(num_classes * test_ratio))
        train_num = num_classes - val_num - test_num
        if train_num <= 0:
            raise ValueError(
                f"Ratio class split leaves no training classes: C={num_classes}, "
                f"val_ratio={val_ratio}, test_ratio={test_ratio}."
            )
        requested = num_classes

    train_classes = ordered_classes[:train_num]
    val_classes = ordered_classes[train_num : train_num + val_num]
    test_classes = ordered_classes[train_num + val_num : train_num + val_num + test_num]
    unused_classes = ordered_classes[train_num + val_num + test_num :]

    if fs_class_split is not None and unused_classes:
        warnings.warn(
            f"Explicit class split uses {requested}/{num_classes} classes; "
            f"unused classes are dropped from few-shot experiments: {unused_classes}",
            stacklevel=2,
        )

    _assert_disjoint(train_classes, val_classes, test_classes)
    if not train_classes or not val_classes or not test_classes:
        raise ValueError(
            "Few-shot class split produced an empty train/val/test split. "
            "Adjust --fs_class_split or class ratios."
        )

    print(
        "Few-shot class split counts: "
        f"train={len(train_classes)}, val={len(val_classes)}, "
        f"test={len(test_classes)}, unused={len(unused_classes)}"
    )

    return train_classes, val_classes, test_classes, unused_classes


def build_class_to_nodes(y):
    """Build a mapping from original class id to original node indices."""
    labels = y.detach().cpu().view(-1)
    class_to_nodes = {}
    for cls in torch.unique(labels, sorted=True).tolist():
        cls_id = int(cls)
        class_to_nodes[cls_id] = (labels == cls_id).nonzero(as_tuple=False).view(-1).long()
    return class_to_nodes


def build_fewshot_splits(args, data, seed):
    """Build auditable class-disjoint few-shot splits for one seed."""
    y_cpu = data.y.detach().cpu()
    original_classes = [int(cls) for cls in torch.unique(y_cpu, sorted=True).tolist()]
    train_classes, val_classes, test_classes, unused_classes = split_classes(
        y=y_cpu,
        seed=seed,
        fs_class_split=args.fs_class_split,
        val_ratio=args.fs_val_class_ratio,
        test_ratio=args.fs_test_class_ratio,
        fixed_order=args.fs_fixed_class_order,
    )
    class_to_nodes = build_class_to_nodes(y_cpu)
    min_nodes = args.fs_shot + args.fs_query
    train_classes, dropped_train = filter_classes_by_min_nodes(class_to_nodes, train_classes, min_nodes)
    val_classes, dropped_val = filter_classes_by_min_nodes(class_to_nodes, val_classes, min_nodes)
    test_classes, dropped_test = filter_classes_by_min_nodes(class_to_nodes, test_classes, min_nodes)

    train_way = args.fs_train_way or args.fs_way
    val_way = args.fs_val_way or args.fs_way
    test_way = args.fs_test_way or args.fs_way
    _validate_disjoint_and_way(train_classes, val_classes, test_classes, train_way, val_way, test_way)

    split_dict = {
        "seed": int(seed),
        "original_classes": original_classes,
        "train_classes": train_classes,
        "val_classes": val_classes,
        "test_classes": test_classes,
        "unused_classes": unused_classes,
        "dropped_classes": {
            "train": dropped_train,
            "val": dropped_val,
            "test": dropped_test,
        },
    }
    split_dict["split_hash"] = split_hash(split_dict)
    return split_dict


def filter_classes_by_min_nodes(class_to_nodes, classes, min_nodes):
    """Drop classes that cannot provide one support/query sample set."""
    filtered_classes = []
    dropped_classes = []

    for cls in classes:
        cls_id = int(cls)
        node_count = int(class_to_nodes.get(cls_id, torch.empty(0)).numel())
        if node_count < min_nodes:
            dropped_classes.append(cls_id)
        else:
            filtered_classes.append(cls_id)

    if dropped_classes:
        warnings.warn(
            f"Dropping classes with fewer than {min_nodes} nodes: {dropped_classes}",
            stacklevel=2,
        )

    return filtered_classes, dropped_classes


def sample_episode(class_to_nodes, candidate_classes, way, shot, query, device=None, generator=None):
    """Sample one N-way K-shot Q-query episode from candidate classes."""
    if way <= 0 or shot <= 0 or query <= 0:
        raise ValueError("Episode way, shot, and query must all be positive.")
    if len(candidate_classes) < way:
        raise ValueError(
            f"Cannot sample {way}-way episode from only {len(candidate_classes)} candidate classes."
        )

    class_perm = torch.randperm(len(candidate_classes), generator=generator)[:way].tolist()
    selected_classes = [int(candidate_classes[idx]) for idx in class_perm]

    support_indices = []
    query_indices = []
    support_labels = []
    query_labels = []
    needed_nodes = shot + query

    for local_label, cls_id in enumerate(selected_classes):
        nodes = class_to_nodes[cls_id]
        if nodes.numel() < needed_nodes:
            raise ValueError(
                f"Class {cls_id} has {nodes.numel()} nodes, but an episode needs "
                f"{needed_nodes} nodes ({shot} support + {query} query)."
            )

        node_perm = torch.randperm(nodes.numel(), generator=generator)[:needed_nodes]
        sampled_nodes = nodes[node_perm]
        support_nodes = sampled_nodes[:shot]
        query_nodes = sampled_nodes[shot:]

        support_indices.append(support_nodes)
        query_indices.append(query_nodes)
        support_labels.append(torch.full((shot,), local_label, dtype=torch.long))
        query_labels.append(torch.full((query,), local_label, dtype=torch.long))

    support_idx = torch.cat(support_indices, dim=0).long()
    query_idx = torch.cat(query_indices, dim=0).long()
    support_y = torch.cat(support_labels, dim=0).long()
    query_y = torch.cat(query_labels, dim=0).long()

    if set(support_idx.tolist()) & set(query_idx.tolist()):
        raise RuntimeError("Invalid episode: support_idx and query_idx overlap.")

    if device is not None:
        support_idx = support_idx.to(device)
        query_idx = query_idx.to(device)
        support_y = support_y.to(device)
        query_y = query_y.to(device)

    return {
        "classes": selected_classes,
        "support_idx": support_idx,
        "query_idx": query_idx,
        "support_y": support_y,
        "query_y": query_y,
    }


def build_episode_bank(class_to_nodes, split_classes, way, shot, query, num_episodes, seed):
    """Build a deterministic CPU episode bank from one split."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive when building an episode bank.")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    episodes = []
    for _ in range(int(num_episodes)):
        episodes.append(
            sample_episode(
                class_to_nodes,
                split_classes,
                way,
                shot,
                query,
                device=None,
                generator=generator,
            )
        )
    return episodes


def build_all_episode_banks(args, data, split_dict, seed):
    """Load or build train/val/test episode banks for one seed."""
    class_to_nodes = build_class_to_nodes(data.y.detach().cpu())
    bank_specs = {
        "train": {
            "classes": split_dict["train_classes"],
            "way": args.fs_train_way or args.fs_way,
            "num_episodes": args.fs_train_episodes,
            "seed": _episode_seed(seed, 11),
        },
        "val": {
            "classes": split_dict["val_classes"],
            "way": args.fs_val_way or args.fs_way,
            "num_episodes": args.fs_val_episodes,
            "seed": _episode_seed(seed, 23),
        },
        "test": {
            "classes": split_dict["test_classes"],
            "way": args.fs_test_way or args.fs_way,
            "num_episodes": args.fs_test_episodes,
            "seed": _episode_seed(seed, 37),
        },
    }
    paths = episode_bank_paths(args, seed)
    all_paths_exist = all(os.path.exists(path) for path in paths.values())
    episode_banks = {"train": None, "val": None, "test": None}
    episode_hashes = {}

    if args.fs_reuse_episode_bank and all_paths_exist:
        for split_name, spec in bank_specs.items():
            payload = torch.load(paths[split_name], weights_only=False)
            _validate_episode_bank_payload(payload, args, split_dict, split_name, spec)
            episodes = payload["episodes"]
            episode_banks[split_name] = episodes
            episode_hashes[split_name] = payload["meta"]["episode_hash"]
        reused = True
    else:
        reused = False
        for split_name, spec in bank_specs.items():
            episodes = build_episode_bank(
                class_to_nodes,
                spec["classes"],
                spec["way"],
                args.fs_shot,
                args.fs_query,
                spec["num_episodes"],
                spec["seed"],
            )
            episode_hashes[split_name] = episode_bank_hash(episodes)
            episode_banks[split_name] = episodes
            if args.fs_save_episode_bank:
                os.makedirs(os.path.dirname(paths[split_name]), exist_ok=True)
                torch.save(
                    {
                        "split_dict": split_dict,
                        "episodes": episodes,
                        "meta": _episode_bank_meta(args, split_dict, split_name, spec, episode_hashes[split_name]),
                        "fewshot_args": _fewshot_args_dict(args),
                    },
                    paths[split_name],
                )

    episode_banks["meta"] = {
        "reused": reused,
        "paths": paths,
        "episode_hashes": episode_hashes,
        "split_hash": split_dict["split_hash"],
        "specs": {
            split_name: {
                "way": int(spec["way"]),
                "shot": int(args.fs_shot),
                "query": int(args.fs_query),
                "num_episodes": int(spec["num_episodes"]),
                "classes": [int(cls) for cls in spec["classes"]],
            }
            for split_name, spec in bank_specs.items()
        },
    }
    return episode_banks


def episode_bank_paths(args, seed):
    split_tag = _class_split_tag(args)
    prefix = f"{args.dname}_{split_tag}_seed{seed}"
    return {
        "train": os.path.join(args.fs_episode_bank_dir, f"{prefix}_train.pt"),
        "val": os.path.join(args.fs_episode_bank_dir, f"{prefix}_val.pt"),
        "test": os.path.join(args.fs_episode_bank_dir, f"{prefix}_test.pt"),
    }


def split_hash(split_dict):
    payload = {
        "seed": int(split_dict["seed"]),
        "original_classes": split_dict["original_classes"],
        "train_classes": split_dict["train_classes"],
        "val_classes": split_dict["val_classes"],
        "test_classes": split_dict["test_classes"],
        "unused_classes": split_dict["unused_classes"],
        "dropped_classes": split_dict["dropped_classes"],
    }
    return _stable_hash(payload)


def episode_bank_hash(episodes):
    payload = []
    for episode in episodes:
        payload.append(
            {
                "classes": [int(cls) for cls in episode["classes"]],
                "support_idx": episode["support_idx"].detach().cpu().tolist(),
                "query_idx": episode["query_idx"].detach().cpu().tolist(),
                "support_y": episode["support_y"].detach().cpu().tolist(),
                "query_y": episode["query_y"].detach().cpu().tolist(),
            }
        )
    return _stable_hash(payload)


def _assert_disjoint(train_classes, val_classes, test_classes):
    train_set = set(train_classes)
    val_set = set(val_classes)
    test_set = set(test_classes)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError(
            "Few-shot class split is invalid: train/val/test class sets must be disjoint."
        )


def _validate_disjoint_and_way(train_classes, val_classes, test_classes, train_way, val_way, test_way):
    _assert_disjoint(train_classes, val_classes, test_classes)
    split_requirements = {
        "train": (len(train_classes), train_way),
        "val": (len(val_classes), val_way),
        "test": (len(test_classes), test_way),
    }
    for split_name, (num_classes, way) in split_requirements.items():
        if num_classes < way:
            raise ValueError(
                f"Few-shot {split_name} split has {num_classes} usable classes, "
                f"but {split_name}_way={way}. Lower the way/shot/query values or adjust --fs_class_split."
            )


def _episode_seed(seed, offset):
    return int(seed) * 10007 + int(offset)


def _class_split_tag(args):
    if args.fs_class_split is not None:
        return args.fs_class_split.replace("/", "-")
    return f"ratio-v{args.fs_val_class_ratio}-t{args.fs_test_class_ratio}"


def _episode_bank_meta(args, split_dict, split_name, spec, episode_hash_value):
    return {
        "dname": args.dname,
        "seed": int(split_dict["seed"]),
        "split_name": split_name,
        "split_hash": split_dict["split_hash"],
        "episode_hash": episode_hash_value,
        "fs_class_split": args.fs_class_split,
        "fs_fixed_class_order": bool(args.fs_fixed_class_order),
        "fs_val_class_ratio": float(args.fs_val_class_ratio),
        "fs_test_class_ratio": float(args.fs_test_class_ratio),
        "way": int(spec["way"]),
        "shot": int(args.fs_shot),
        "query": int(args.fs_query),
        "num_episodes": int(spec["num_episodes"]),
        "episode_seed": int(spec["seed"]),
        "classes": [int(cls) for cls in spec["classes"]],
    }


def _fewshot_args_dict(args):
    return {
        key: getattr(args, key)
        for key in sorted(vars(args))
        if key.startswith("fs_") or key in ["dname", "task_type", "embedding_hidden", "zen_mode"]
    }


def _validate_episode_bank_payload(payload, args, split_dict, split_name, spec):
    if not isinstance(payload, dict) or "episodes" not in payload or "meta" not in payload:
        raise ValueError(f"Episode bank for split {split_name} has an invalid payload format.")
    meta = payload["meta"]
    expected = _episode_bank_meta(args, split_dict, split_name, spec, episode_bank_hash(payload["episodes"]))
    check_keys = [
        "dname",
        "seed",
        "split_name",
        "split_hash",
        "fs_class_split",
        "fs_fixed_class_order",
        "fs_val_class_ratio",
        "fs_test_class_ratio",
        "way",
        "shot",
        "query",
        "num_episodes",
        "classes",
    ]
    mismatches = []
    for key in check_keys:
        if meta.get(key) != expected.get(key):
            mismatches.append((key, meta.get(key), expected.get(key)))
    if meta.get("episode_hash") != expected["episode_hash"]:
        mismatches.append(("episode_hash", meta.get("episode_hash"), expected["episode_hash"]))
    if mismatches:
        mismatch_text = "; ".join(
            f"{key}: loaded={loaded!r}, expected={expected_value!r}"
            for key, loaded, expected_value in mismatches
        )
        raise ValueError(
            f"Existing episode bank for split {split_name} is inconsistent with current few-shot args. "
            f"Set --fs_reuse_episode_bank False to regenerate. Mismatches: {mismatch_text}"
        )


def _stable_hash(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
