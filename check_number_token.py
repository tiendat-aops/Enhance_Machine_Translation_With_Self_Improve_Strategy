from underthesea import word_tokenize

def en_tokenizer(sent: str):
    if len(sent) == 0:
        return []
    return [x for x in sent.split(' ') if x != '']


def vi_tokenizer(sent: str):
    if (len(sent) == 0):
        return []
    return [x for x in word_tokenize(sent) if x != '']

def read_dataset():
    print("Loading datasets...")

    with open('./data/en.txt', 'r') as f:
        en = f.readlines()

    with open('./data/vi.txt', 'r') as f:
        vi = f.readlines()

    en = [s.lower() for s in en]
    vi = [s.lower() for s in vi]
    return en, vi
    # en_train, en_test, vi_train, vi_test = en[:
    #                                           TRAIN_SIZE], en[TRAIN_SIZE:], vi[:TRAIN_SIZE], vi[TRAIN_SIZE:]
    # return en_train, en_test, vi_train, vi_test


def read_masked_dataset():
    with open('/work/dat-nt/Machine_Translation/nmt_custom/data/vi_masked.txt', 'r') as f:
        vi_masked = f.readlines()
    with open('/work/dat-nt/Machine_Translation/nmt_custom/data/vi_label.txt', 'r') as f:
        vi_label = f.readlines()
    with open('/work/dat-nt/Machine_Translation/nmt_custom/data/en_masked.txt', 'r') as f:
        en_masked = f.readlines()
    with open('/work/dat-nt/Machine_Translation/nmt_custom/data/en_label.txt', 'r') as f:
        en_label = f.readlines()
    vi_masked = [vi.lower() for vi in vi_masked]
    en_masked = [en.lower() for en in en_masked]
    vi_label = [vi.lower() for vi in vi_label]
    en_label = [en.lower() for en in en_label]
    return vi_masked, vi_label, en_masked, en_label

en, vi = read_dataset()
#  = read_dataset()
num_tokens = 0
max_token = 0
for e in en:
    num_tokens += len(en_tokenizer(e))
for v in vi:
    num_tokens += len(vi_tokenizer(v))
    max_token = max(max_token, len(vi_tokenizer(v)))

print(num_tokens)
print(max_token)