from lib import *


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in "Attention is All You Need".

    This module adds positional information to the input embeddings to allow
    the model to consider the position of words in a sequence. The positional
    encoding is added to the token embeddings to provide information about
    the relative position of tokens in the sequence.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied after adding positional encoding.
        pos_embedding (torch.Tensor): Precomputed positional encodings.
    """

    def __init__(self, n_embed: int, dropout: int | float):
        """
        Initializes the PositionalEncoding module.

        Args:
            n_embed (int): The dimension of the embeddings.
            dropout (int | float): The dropout rate to apply after adding positional encoding.
        """
        max_len = cfg.max_length
        super().__init__()
        den = torch.exp(-torch.arange(0, n_embed, 2)
                        * math.log(10000) / n_embed)
        pos = torch.arange(0, max_len).reshape(int(max_len), 1)
        pos_embedding = torch.zeros(max_len, n_embed)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        """
        Adds positional encoding to the input token embeddings.

        Args:
            token_embedding (torch.Tensor): The input token embeddings.

        Returns:
            torch.Tensor: The token embeddings with added positional encodings.
        """
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    This module converts token indices into dense vectors of fixed size and scales them
    by the square root of the embedding dimension.

    Attributes:
        embedding (nn.Embedding): The embedding layer.
        n_embed (int): The dimension of the embeddings.
    """

    def __init__(self,
                 vocab_size: int,
                 n_embed: int):
        """
        Initializes the TokenEmbedding module.

        Args:
            vocab_size (int): The size of the vocabulary.
            n_embed (int): The dimension of the embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed, padding_idx=PAD_IDX)
        self.n_embed = n_embed

    def forward(self, tokens):  # (T, B, C)
        """
        Converts token indices to embeddings and scales them.

        Args:
            tokens (torch.Tensor): The input token indices.

        Returns:
            torch.Tensor: The scaled token embeddings.
        """
        # (T, B, C)
        return self.embedding(tokens.long()) * math.sqrt(self.n_embed)
