from lib import *
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("VietAI/envit5-translation")

def yield_tokens(data: Dataset):
    for data_sample in data:
        yield tokenizer.tokenize(data_sample)
# print(tokenizer.tokenize('And of those 10 percent that landed , 16 percent didn &apos;t even go off ; they were duds .'))
sent = '''
Mưa to kéo dài còn khiến tuyến kè tổ 6, khu 1B, phường Hồng Hải bị sạt làm đổ một ngôi nhà, may mắn không gây thiệt hại về người. Chính quyền đã di chuyển các hộ dân trong khu vực sạt lở về nhà văn hóa, chăng dây, đặt biển cảnh báo tại điểm có nguy cơ tiếp tục sạt lở.
'''

# print(tokenizer.tokenize(sent))
# print(tokenizer_en.tokenize(sent))

# with open('/work/dat-nt/Machine_Translation/en-vi-nmt-with-transformer_v0/PhoMT/vi.txt', 'r') as file:
#     data = file.readlines()
# length = []
# count = len(data)
# count_ = 0
# for d in data:
#     l = len(tokenizer.tokenize(d))
#     length.append(l)
#     if l > 600:
#         count_ += 1

# count_ = 0
# count = 0
# for l in length:
#     count += 1
#     if l > 800:
#         count_ += 1

# print(count_)
# print(count_/count)