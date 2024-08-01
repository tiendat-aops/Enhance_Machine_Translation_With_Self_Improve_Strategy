from lib import *
from mask import *
from translate import *


def evaluate_model(model, val_data, batch_size, collate_fn, loss_fn):
    """
    Evaluates the model on the validation dataset.

    Args:
        model (nn.Module): The model to be evaluated.
        val_data (Dataset): The validation dataset.
        batch_size (int): The batch size to be used for evaluation.
        collate_fn (function): Function to collate data samples into batch tensors.
        loss_fn (function): Loss function to be used during evaluation.

    Returns:
        float: The average loss over the validation dataset.
    """
    model.eval()
    losses = 0

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        generator=torch.Generator(device='cuda')
    )
    total = math.ceil(len(val_data) / batch_size)

    with torch.no_grad():
        for i, (src, tgt) in tqdm(enumerate(val_dataloader), total=total, dynamic_ncols=True):
            with torch.autocast(device_type=str(DEVICE), dtype=torch.bfloat16):
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)

                tgt_input = tgt[:-1, :]  # (T, B)

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                    src, tgt_input)

                logits = model(src, tgt_input, src_mask, tgt_mask,
                               src_padding_mask, tgt_padding_mask, src_padding_mask)

                tgt_out = tgt[1:, :].type(torch.long)  # (T, B)
                loss = loss_fn(
                    logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                losses += loss.item()

    torch.cuda.empty_cache()
    gc.collect()

    return losses / len(list(val_dataloader))


def calculate_bleu_greedy(model, source, target, text_transform, vocab_transform):
    print("Calculating BLEU score with greedy translate...")
    model.eval()
    pred_greedy = []
    for sent in tqdm(source, dynamic_ncols=True):
        text = translate_greedy(model, sent, text_transform, vocab_transform)
        # print(text)
        # text = 'Nó giúp chúng ta chỉ ném bom những gì mà chúng ta hoàn toàn cần phải huỷ diệt'
        text = text.split(' ')
        text = md.detokenize(text)
        pred_greedy.append(text)
    ref = [[md.detokenize(s.split(' ')) for s in target]]

    bleu = BLEU()
    res = bleu.corpus_score(pred_greedy, ref)
    return res
