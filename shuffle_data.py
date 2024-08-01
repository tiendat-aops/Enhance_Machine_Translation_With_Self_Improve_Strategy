# import torch
# VOCAB_PATH= "/work/dat-nt/Machine_Translation/nmt_custom/vocab/mt5_24_07.pth"
# vocab = torch.load(VOCAB_PATH)
# vocab_transform = vocab['vocab_transform']
# print(vocab_transform.__getitem__('<eos>'))

# import nltk
# from nltk.translate.bleu_score import corpus_bleu

# def calculate_bleu_score(reference_sentences, predicted_sentences):
#     """
#     Calculates the BLEU score for the given reference and predicted sentences.

#     Args:
#         reference_sentences (list of list of str): List of reference sentences, where each reference sentence is split into a list of words.
#         predicted_sentences (list of list of str): List of predicted sentences, where each predicted sentence is split into a list of words.

#     Returns:
#         float: The BLEU score.
#     """
#     # Ensure the reference sentences are in the correct format
#     reference_list = [[ref.split()] for ref in reference_sentences]
#     print(reference_list)
#     # Ensure the predicted sentences are in the correct format  
#     predicted_list = [pred.split() for pred in predicted_sentences]
#     print(predicted_list)
#     # Calculate the BLEU score
#     bleu_score = corpus_bleu(reference_list, predicted_list)
    
#     return bleu_score

# # Example usage
# reference_sentences = [
#     "the cat is on the mat",
#     "there is a cat on the mat"
# ]

# predicted_sentences = [
#     "the cat is on the mat",
#     "there is a cat on the mat"
# ]

# bleu_score = calculate_bleu_score(reference_sentences, predicted_sentences)
# print(f"BLEU score: {bleu_score:.4f}")

from sklearn.model_selection import train_test_split

x = [1,2,3,4,5,6,6,7,8,9]
y = [0,0,0,0,0,1,1,1,1,1]
a_0,a_1,b_0,b_1 = train_test_split(x, y, test_size=3, shuffle=True)
print(a_0)
print(a_1)
print(b_0)
print(b_1)