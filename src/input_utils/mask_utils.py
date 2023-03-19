import torch

def generate_mask_input(
    freq_x,
    patch_resolution=(1, 1),
    window_size=(1, 1),
    mask_ratio=0.75,
):
    """
    Optimized window masking: generate patch masks
    """
    
    B = freq_x.shape[0]

    ph, pw = patch_resolution[0], patch_resolution[1]  # num patches h and w
    dh, dw = int(ph // window_size[0]), int(pw // window_size[1])  # window_resolution h and w

    rh, rw = window_size[0], window_size[1]
    
    
    # random mask [b, window_resolution height, window_resolution_width]
    bit_mask = torch.cuda.FloatTensor(B, dh, dw).uniform_() > mask_ratio
    
    # [b, patch_resolution_height, window_resolution_width]
    patch_mask = bit_mask.repeat_interleave(rh, dim=1)
    
    # [b, patch_resolution_height, patch_resolution_width]
    patch_mask = patch_mask.repeat_interleave(rw, dim=2)
    
    # [b, patch_resolutions]
    patch_mask = patch_mask.reshape(B, -1).int().float()
    
    return patch_mask

def mask_input(
    freq_x,
    input_resolution,
    patch_resolution=(1, 1),
    channel_dimension=-1,
    window_size=(1, 1),
    mask_token=None,
    mask_ratio=0.75,
):
    """
    Optimized window masking: get masks and apply to inputs
    """
    if len(freq_x.shape) > 3:
        """DeepSense [B, c, i, s] -> [B, i * s, c]"""
        b, c, i, s = freq_x.shape
        x = freq_x.reshape(b, c, -1).permute(0, 2, 1)
    else:
        x = freq_x

    B, L, D = x.shape
    
    # generate masks
    patch_mask = generate_mask_input(freq_x, patch_resolution, window_size, mask_ratio)

    channel_repeat = [1, 1, 1]
    channel_repeat[channel_dimension] = D # (1, 1, D) or (1, D, 1)
    
    # [b, patch_resolution, D] or [b, D, patch_resolution]
    patch_mask_channel = patch_mask.unsqueeze(channel_dimension).repeat(channel_repeat)
    
    # mask_token -> [b, D] -> [b, 1, D]
    mask_token = mask_token.unsqueeze(0).repeat(B, 1)
    mask_token = mask_token.unsqueeze(1)

    # masked [b, patch_resolution, 1]
    token_mask = (1 - patch_mask).unsqueeze(-1)
    
    # [b, patch_resolution, 1] @ [b, 1, D] -> [b, patch_resolution, D]
    # token * (1 - mask)
    tokens = torch.bmm(token_mask, mask_token)
    # x * mask
    masked_input = x * patch_mask_channel
    # x * mask + token * (1 - mask)
    masked_x = masked_input + tokens
    
    return masked_x, patch_mask.int()
    