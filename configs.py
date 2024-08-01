import argparse
from dataclasses import dataclass

@dataclass
class Config:
    src_language: str
    tgt_language: str
    data_name: str
    emb_size: int
    nhead: int
    ffn_hid_dim: int
    max_length: int
    batch_size: int
    num_encoder_layers: int
    num_decoder_layers: int
    learning_rate: float
    num_epochs: int
    num_phrases: int
    dropout: float
    activation: str
    train_size: int
    test_size: int
    beam_size: int
    warm_up: int
    load_checkpoint: bool
    use_masked_dataset: bool
    number_vi_masked: int
    number_en_masked: int
    reverse_translate: bool
    train_from_scratch: bool
    folder_path: str
    vocab_path: str
    save_model_path: str
    checkpoint_path: str
    self_improve: bool

    def show_config(self):
        print('Model configs:')
        for k, v in vars(self).items():
            print(f'- {k:<20}: {v}')
    
    def get_config(self):
        res = 'Model configs:'
        for k, v in vars(self).items():
            res += f'\n- {k:<20}: {v}'
        return res