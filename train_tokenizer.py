from lib import *

abs_path = Path(__file__).parent
data_path = os.path.join(abs_path, 'data')
# print(abs_path)
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


def prepare_tokenizer_trainer(alg):
    """
    Prepares the tokenizer and trainer based on the specified algorithm.

    This function initializes the tokenizer and its trainer based on the
    specified algorithm. Currently, it supports Byte Pair Encoding (BPE).

    Args:
        alg (str): The algorithm to use for tokenization. Supported value is 'BPE'.

    Returns:
        Tuple[Tokenizer, Trainer]: A tuple containing the tokenizer and the trainer.
    """
    if alg == 'BPE':
        tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(special_tokens=special_symbols)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    return tokenizer, trainer


def train_tokenizer(files, alg='BPE', lang=''):
    """
    Trains the tokenizer on the provided files.

    This function trains the tokenizer on the specified files using the specified
    algorithm. After training, the tokenizer is saved to a file and reloaded.

    Args:
        files (List[str]): A list of file paths to train the tokenizer on.
        alg (str, optional): The algorithm to use for tokenization. Default is 'BPE'.
        lang (str, optional): The language code to include in the saved tokenizer filename.

    Returns:
        Tokenizer: The trained tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer(alg)
    tokenizer.train(files, trainer)
    tokenizer.save(f"{abs_path}/tokenizer/{alg}_tokenizer_{lang}.json")
    tokenizer = Tokenizer.from_file(
        f"{abs_path}/tokenizer/{alg}_tokenizer_{lang}.json")
    return tokenizer


def main():
    alg = 'BPE'
    list_files = os.listdir(data_path)
    for file in list_files:
        file_name = file.split('/')[-1].split('.')[0]
        if file_name == 'vi' or 'en':
            train_tokenizer([os.path.join(data_path, file)],
                            alg, lang=file_name)


if __name__ == '__main__':
    # print(abs_path)
    main()
