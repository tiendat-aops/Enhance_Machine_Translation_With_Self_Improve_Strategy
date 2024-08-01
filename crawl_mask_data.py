from datasets import load_dataset
import random
from tqdm import tqdm
# ds = load_dataset("wanhin/VinAI_PhoMT")

# vi_text = ds['train']['vi']
# random.shuffle(vi_text)

def mask_sentence(sentence, span_length=2, corruption_rate=0.15):
    words = sentence.split()
    num_words = len(words)
    masked_indices = set()
    
    # Calculate the number of words to mask based on the corruption rate
    total_words_to_mask = int(corruption_rate * num_words)
    
    # Calculate the number of spans
    num_spans = total_words_to_mask // span_length
    if num_spans == 0:
        num_spans += 1
    for _ in range(num_spans):
        try:
            start_index = random.randint(0, num_words - span_length)
        except:
            start_index = 0
        
        # Ensure we don't overlap spans
        while any(i in masked_indices for i in range(start_index, start_index + span_length)):
                start_index = random.randint(0, num_words - span_length)
        
        # Mask the span
        for i in range(span_length):
            masked_indices.add(start_index + i)
    
    masked_sentence = []
    for i in range(num_words):
        if i in masked_indices:
            if i == 0 or (i > 0 and words[i-1] != '<M>'):
                masked_sentence.append('<M>')
        else:
            masked_sentence.append(words[i])
    
    return ' '.join(masked_sentence)

def main():
    ds = load_dataset("wanhin/VinAI_PhoMT")
    number_samples = 50000
    text = ds['train']['en']
    random.shuffle(text)
    file_masked = open('/work/dat-nt/Machine_Translation/nmt_custom/data/en_masked.txt', 'a')
    file_label = open('/work/dat-nt/Machine_Translation/nmt_custom/data/en_label.txt', 'a')
    for txt in tqdm(text[:number_samples]):
        masked_sentence = mask_sentence(txt, span_length=2, corruption_rate=0.15)
        file_masked.write(f"{masked_sentence}\n")
        file_label.write(txt)
    file_masked.close()
    file_label.close()

if __name__ == '__main__':
    main()