from lib import *
from mask import *


def greedy_decode(model: nn.Module, src, src_mask, max_len, start_symbol, **kwargs):
    """
    Generate output sequence using greedy algorithm.

    Args:
        model (nn.Module): The model to use for decoding.
        src (Tensor): The source tensor.
        src_mask (Tensor): The source mask tensor.
        max_len (int): The maximum length of the generated sequence.
        start_symbol (int): The index of the start symbol.
        **kwargs: Additional arguments.

    Returns:
        Tensor: The generated output sequence.
    """
    model.eval()
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(
        DEVICE)  # ys is gonna have shape (T, B) with B = 1 when generating
    with torch.no_grad():
        for i in range(max_len-1):
            memory = memory.to(DEVICE)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)  # (T, B) -> (B, T)
            prob = model.generator(out[:, -1])
            _, next_word_idx = torch.max(prob, dim=1)
            next_word_idx = next_word_idx.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word_idx)], dim=0)
            if next_word_idx == EOS_IDX:
                break
    return ys


def translate_greedy(model: nn.Module, src_sentence: str, text_transform, vocab_transform):
    """
    Translates a source sentence using a greedy decoding strategy.

    Args:
        model (nn.Module): The translation model to be used.
        src_sentence (str): The source sentence to translate.
        text_transform (function): Function to transform words to IDs.
        vocab_transform (Vocab): Vocabulary object for converting IDs back to words.

    Returns:
        str: The translated sentence.
    """
    model.eval()

    src_sentence = mt.tokenize(src_sentence, return_str=True)
    with torch.no_grad():
        src = text_transform(src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens,
                    device=DEVICE)).type(torch.bool)
        tgt_tokens = greedy_decode(model=model, src=src, src_mask=src_mask, max_len=int(
            1.6 * num_tokens), start_symbol=BOS_IDX).flatten()

    result = "".join(vocab_transform.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace(
        "<bos>", "").replace("<eos>", "").replace("<pad>", "").replace("<unk>", "").replace("##", " ").replace('▁', ' ')[1:]
    if cfg.self_improve:
        with open(f"{cfg.folder_path}/log/result_multi_phrase.log", 'a') as log:
            log.write(result)
            log.write('\n')
            log.close()
    else:
        with open(f"result.log", 'a') as log:
            log.write(result)
            log.write('\n')
            log.close()
    return result

def beam_search_decode(model: nn.Module, src, src_mask, max_len, start_symbol, beam_size=3, **kwargs):
    """
    Generate output sequence using beam search algorithm.

    Args:
        model (nn.Module): The model to use for decoding.
        src (Tensor): The source tensor.
        src_mask (Tensor): The source mask tensor.
        max_len (int): The maximum length of the generated sequence.
        start_symbol (int): The index of the start symbol.
        beam_size (int): The beam size for beam search.
        **kwargs: Additional arguments.

    Returns:
        Tensor: The generated output sequence.
    """
    model.eval()
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    
    # Hypotheses: list of (log probability, sequence) tuples
    hypotheses = [(0, torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE))]
    
    completed_hypotheses = []
    
    with torch.no_grad():
        for i in range(max_len - 1):
            new_hypotheses = []
            for log_prob, seq in hypotheses:
                tgt_mask = (generate_square_subsequent_mask(seq.size(0)).type(torch.bool)).to(DEVICE)
                memory = memory.to(DEVICE)
                out = model.decode(seq, memory, tgt_mask)
                out = out.transpose(0, 1)  # (T, B) -> (B, T)
                prob = model.generator(out[:, -1])
                log_probabilities = F.log_softmax(prob, dim=1)
                
                topk_log_probs, topk_idxs = log_probabilities.topk(beam_size)
                
                for j in range(beam_size):
                    new_seq = torch.cat([seq, torch.ones(1, 1).type_as(src.data).fill_(topk_idxs[0][j])], dim=0)
                    new_log_prob = log_prob + topk_log_probs[0][j].item()
                    new_hypotheses.append((new_log_prob, new_seq))
                    
                    if topk_idxs[0][j] == EOS_IDX:
                        completed_hypotheses.append((new_log_prob, new_seq))
            
            # Keep the top `beam_size` hypotheses
            hypotheses = heapq.nlargest(beam_size, new_hypotheses, key=lambda x: x[0])
            
            # If all hypotheses end with EOS_IDX, stop early
            if all(seq[-1].item() == EOS_IDX for _, seq in hypotheses):
                break
    
    if len(completed_hypotheses) == 0:
        completed_hypotheses = hypotheses
    
    # Return the hypothesis with the highest log probability
    best_hypothesis = max(completed_hypotheses, key=lambda x: x[0])
    return best_hypothesis[1]

# Example usage
# ys = beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size=3)


def translate_beam_search(model: nn.Module, src_sentence: str, text_transform, vocab_transform, beam_size=3):
    """
    Translates a source sentence using a beam search decoding strategy.

    Args:
        model (nn.Module): The translation model to be used.
        src_sentence (str): The source sentence to translate.
        text_transform (function): Function to transform words to IDs.
        vocab_transform (Vocab): Vocabulary object for converting IDs back to words.
        beam_size (int): The beam size for beam search.

    Returns:
        str: The translated sentence.
    """
    model.eval()

    src_sentence = mt.tokenize(src_sentence, return_str=True)
    with torch.no_grad():
        src = text_transform(src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens, device=DEVICE)).type(torch.bool)
        tgt_tokens = beam_search_decode(model=model, src=src, src_mask=src_mask, max_len=int(
            1.6 * num_tokens), start_symbol=BOS_IDX, beam_size=beam_size).flatten()

    result = "".join(vocab_transform.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace(
        "<bos>", "").replace("<eos>", "").replace("<pad>", "").replace("<unk>", "").replace("##", " ").replace('▁', ' ')[1:]
    with open("result.log", 'a') as log:
        log.write(result)
        log.write('\n')
        log.close()
    return result
