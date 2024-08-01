from lib import *


def generate_square_subsequent_mask(sz):
    """
    Generates a square mask for the sequence to prevent attending to future tokens.

    The mask is used to ensure that each position only attends to the previous
    positions (including the current one) in the sequence, which is necessary for
    autoregressive models like the transformer decoder during training.

    Args:
        sz (int): The size of the mask (sequence length).

    Returns:
        torch.Tensor: A square mask tensor of shape (sz, sz) with float values.
    """
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    """
    Creates masks for source and target sequences for transformer models.

    This function generates the necessary masks for the source and target sequences
    to be used in the transformer model. These include the source mask, target mask,
    and padding masks for both source and target sequences.

    Args:
        src (torch.Tensor): Source input sequence tensor of shape (src_seq_len, batch_size).
        tgt (torch.Tensor): Target input sequence tensor of shape (tgt_seq_len, batch_size).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - src_mask (torch.Tensor): The source mask of shape (src_seq_len, src_seq_len).
            - tgt_mask (torch.Tensor): The target mask of shape (tgt_seq_len, tgt_seq_len).
            - src_padding_mask (torch.Tensor): The source padding mask of shape (batch_size, src_seq_len).
            - tgt_padding_mask (torch.Tensor): The target padding mask of shape (batch_size, tgt_seq_len).
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # We are predicting the future so past tokens cannot communicate with future tokens
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # We already know every tokens about the src => no need to mask the future
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(
        0, 1)  # boolean tensor (B, T, C) -> (T, B, C)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(
        0, 1)  # boolean tensor (B, T, C) -> (T, B, C)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
