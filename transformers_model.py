from lib import *
from embeddings import *


class Seq2SeqTransformer(nn.Module):
    """
    Sequence-to-Sequence Transformer model.

    This model consists of an encoder and a decoder that use the transformer
    architecture to perform sequence-to-sequence tasks such as translation.

    Attributes:
        transformer (nn.Transformer): The core transformer model.
        _generator (nn.Linear): Linear layer for generating final output from decoder.
        src_tok_emb (TokenEmbedding): Token embedding layer for the source language.
        tgt_tok_emb (TokenEmbedding): Token embedding layer for the target language.
        positional_encoding (PositionalEncoding): Positional encoding layer.
    """

    def __init__(
            self,
            n_encoder_layer: int,
            n_decoder_layer: int,
            n_embed: int,
            n_head: int,
            src_vocab_size: int,
            tgt_vocab_size: int,
            dff: int,
            dropout: float,
            activation: str,
    ):
        """
        Initializes the Seq2SeqTransformer model.

        Args:
            n_encoder_layer (int): Number of encoder layers.
            n_decoder_layer (int): Number of decoder layers.
            n_embed (int): Dimensionality of the embeddings.
            n_head (int): Number of attention heads.
            src_vocab_size (int): Vocabulary size of the source language.
            tgt_vocab_size (int): Vocabulary size of the target language.
            dff (int): Dimensionality of the feedforward network.
            dropout (float): Dropout rate.
            activation (str): Activation function to use.
        """
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=n_embed,
            nhead=n_head,
            num_encoder_layers=n_encoder_layer,
            num_decoder_layers=n_decoder_layer,
            dim_feedforward=dff,
            dropout=dropout,
            activation=activation,
        )
        self._generator = nn.Linear(n_embed, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, n_embed)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, n_embed)
        self.positional_encoding = PositionalEncoding(
            n_embed=n_embed, dropout=0.1)

    def generator(self, x):
        """
        Generates the final output from the decoder.

        Args:
            x (torch.Tensor): The output from the transformer decoder.

        Returns:
            torch.Tensor: The final output logits.
        """
        return self._generator(x)

    def forward(
            self,
            src,
            tgt,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
    ):
        """
        Forward pass for the Seq2SeqTransformer model.

        Args:
            src (torch.Tensor): Source input sequence.
            tgt (torch.Tensor): Target input sequence.
            src_mask (torch.Tensor): Source sequence mask.
            tgt_mask (torch.Tensor): Target sequence mask.
            src_padding_mask (torch.Tensor): Source padding mask.
            tgt_padding_mask (torch.Tensor): Target padding mask.
            memory_key_padding_mask (torch.Tensor): Memory key padding mask.

        Returns:
            torch.Tensor: The output of the model.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))  # (T, B, C)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))  # (T, B, C)
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src, src_mask):
        """
        Encodes the source sequence.

        Args:
            src (torch.Tensor): Source input sequence.
            src_mask (torch.Tensor): Source sequence mask.

        Returns:
            torch.Tensor: The encoded representation of the source sequence.
        """
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        """
        Decodes the target sequence using the encoded memory.

        Args:
            tgt (torch.Tensor): Target input sequence.
            memory (torch.Tensor): Encoded memory from the encoder.
            tgt_mask (torch.Tensor): Target sequence mask.

        Returns:
            torch.Tensor: The decoded representation of the target sequence.
        """
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)
