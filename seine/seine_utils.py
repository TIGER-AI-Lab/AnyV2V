import torch


# Adopt from SEINE/utils.py
def mask_generation_before(mask_type, shape, dtype, device, dropout_prob=0.0, use_image_num=0):
    b, f, c, h, w = shape
    if mask_type.startswith("first"):
        num = int(mask_type.split("first")[-1])
        mask_f = torch.cat(
            [
                torch.zeros(1, num, 1, 1, 1, dtype=dtype, device=device),
                torch.ones(1, f - num, 1, 1, 1, dtype=dtype, device=device),
            ],
            dim=1,
        )
        mask = mask_f.expand(b, -1, c, h, w)
    elif mask_type.startswith("all"):
        mask = torch.ones(b, f, c, h, w, dtype=dtype, device=device)
    elif mask_type.startswith("onelast"):
        num = int(mask_type.split("onelast")[-1])
        mask_one = torch.zeros(1, 1, 1, 1, 1, dtype=dtype, device=device)
        mask_mid = torch.ones(1, f - 2 * num, 1, 1, 1, dtype=dtype, device=device)
        mask_last = torch.zeros_like(mask_one)
        mask = torch.cat([mask_one] * num + [mask_mid] + [mask_last] * num, dim=1)
        mask = mask.expand(b, -1, c, h, w)
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")
    return mask

