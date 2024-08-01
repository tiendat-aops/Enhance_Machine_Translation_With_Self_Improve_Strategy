
from lib import *

class MTDataset(Dataset):
    def __init__(self, src: List[str], tgt: List[str], split: Literal[f"train", "val", 'test']):
        super().__init__()
        self.split = split
        self.X = src
        self.y = tgt

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

    def __len__(self):
        return len(self.X)
    
def shuffle_data(train, test):
    combined = list(zip(train, test))
    random.shuffle(combined)
    shuffled_train, shuffled_test = zip(*combined)
    return list(shuffled_train), list(shuffled_test)

def clean(sent):
    sent = sent.rstrip("\n")
    for pattern, repl in re_clean_patterns:
        sent = re.sub(pattern, repl, sent)
    return sent

def read_dataset(data_name):
    print("Loading datasets...")

    with open(f"/{data_name}/en.txt", 'r') as f:
        en = f.readlines()

    with open(f"/{data_name}/vi.txt", 'r') as f:
        vi = f.readlines()

    all_data = en + vi
    TRAIN_SIZE_ = len(en) - 1000
    en_train, en_test, vi_train,  vi_test = en[:
                                               int(TRAIN_SIZE_)],  en[TRAIN_SIZE_:], vi[:TRAIN_SIZE_], vi[TRAIN_SIZE_:]
    return all_data, en_train, en_test, vi_train, vi_test


def read_masked_dataset():
    with open('./data/vi_masked.txt', 'r') as f:
        vi_masked = f.readlines()
    with open('./data/vi_label.txt', 'r') as f:
        vi_label = f.readlines()
    with open('./data/en_masked.txt', 'r') as f:
        en_masked = f.readlines()
    with open('./data/en_label.txt', 'r') as f:
        en_label = f.readlines()
    vi_masked = [
        f"fill in the blanks: {text_normalize(vi.lower())}" for vi in vi_masked]
    en_masked = [f"fill in the blanks: {clean(en.lower())}" for en in en_masked]
    vi_label = [text_normalize(vi.lower()) for vi in vi_label]
    en_label = [clean(en.lower()) for en in en_label]
    vi_masked, vi_label = shuffle_data(vi_masked, vi_label)
    en_masked, en_label = shuffle_data(en_masked, en_label)
    return vi_masked, vi_label, en_masked, en_label