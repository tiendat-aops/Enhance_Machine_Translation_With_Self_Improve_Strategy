from lib import *
from load_tokenizer import *
from embeddings import *
from transformers_model import *
from mask import *
from translate import *
from evaluate_model import *
from trainer import *
from create_dataset import *


def add_special_tokens(text: List[str]):
    return ['<bos>'] + text + ['<eos>']


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.as_tensor(token_ids)


def build_vocab(vocab_path, data, min_freq, max_tokens):
    if os.path.exists(vocab_path):
        vocab = torch.load(vocab_path)
        vocab_transform = vocab['vocab_transform']
        VOCAB_SIZE = vocab['VOCAB_SIZE']
    else:
        vocab_transform = {}
        print("Building vocabs...")

        vocab_transform = build_vocab_from_iterator(yield_tokens(data),
                                                    min_freq=min_freq,
                                                    specials=special_symbols,
                                                    special_first=True,
                                                    max_tokens=max_tokens)
        vocab_transform.set_default_index(UNK_IDX)

        VOCAB_SIZE = len(vocab_transform)

        torch.save({
            'vocab_transform': vocab_transform,
            "VOCAB_SIZE": VOCAB_SIZE,
        }, vocab_path)
    return vocab_transform, VOCAB_SIZE

def load_model(MODEL_PATH):
    checkpoints = torch.load(MODEL_PATH)
    for k,v in checkpoints['model_state_dict'].items():
        print(k, v.shape)


def main():
    """
    Load the dataset, and split it into train and test sets
    """
    data, en_train, en_test, vi_train, vi_test = read_dataset(cfg.data_name)
    if cfg.reverse_translate:
        src_train = en_train + vi_train
        tgt_train = vi_train + en_train
        en_train, vi_train = shuffle_data(src_train, tgt_train)
    if cfg.use_masked_dataset:
        vi_masked, vi_label, en_masked, en_label = read_masked_dataset()
        en_train = en_train + vi_masked[:cfg.number_vi_masked] + en_masked[:cfg.number_en_masked]
        vi_train = vi_train + vi_label[:cfg.number_vi_masked] + en_label[:cfg.number_en_masked]

    # Split data for training
    src_train_final, val_src, tgt_train_final, val_tgt = train_test_split(en_train, vi_train, test_size=1000, shuffle=True)


    train_data = MTDataset(src_train_final, tgt_train_final, split='train')
    val_data = MTDataset(val_src, val_tgt, split='val')

    """
    Build vocabs for src and tgt languages
    """
    vocab_transform, VOCAB_SIZE = build_vocab(
        cfg.vocab_path, data, min_freq=3, max_tokens=60000)

    text_transform = sequential_transforms(tokenizer.tokenize,
                                           add_special_tokens,
                                           vocab_transform,
                                           tensor_transform)    # Add BOS/EOS and create tensor
    print(f"Vocab size: {VOCAB_SIZE}")
    print(text_transform("hôm nay là thứ bảy"))

    def collate_fn(batch):
        # Certified
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform(src_sample))
            tgt_batch.append(text_transform(tgt_sample))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch
    
    process = 'train'
    if process == 'train':
        print("Start training...")
        if cfg.self_improve:
            train(num_epochs=cfg.num_epochs, train_data=train_data,
                val_data=val_data, en_test=en_test, vi_test=vi_test, collate_fn=collate_fn,
                text_transform=text_transform, vocab_transform=vocab_transform, VOCAB_SIZE=VOCAB_SIZE)
        wandb.finish()
    else:
        en_test_bleu = ['I &apos;ll give you one last illustration of variability , and that is -- oh , I &apos;m sorry .']
        print(clean(en_test_bleu[0]))
if __name__ == '__main__':
    main()
    # read_dataset()
    # load_model('/work/dat-nt/Machine_Translation/nmt_custom/model/model_IWSLT15_training_mt5_tokenizer_v3.pth')
    # print("pass")
