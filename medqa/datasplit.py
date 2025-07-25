import hashlib


def get_split(id_value, seed=42):
    """Compute train/val/test split for a given ID using hash trick (80/10/10)
    Returns: "train", "val", or "test"
    """
    hash_input = f"{id_value}_{seed}".encode("utf-8")
    hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)

    target_split = hash_value % 10

    if target_split == 0:
        return "test"
    elif target_split == 1:
        return "val"
    else:
        return "train"
