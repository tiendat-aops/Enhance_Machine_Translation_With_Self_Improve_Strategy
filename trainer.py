from lib import *
from mask import *
from evaluate_model import *
from transformers_model import *
from create_dataset import *

def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, train_data, collate_fn, loss_fn, grad_accum_steps=5):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        train_data (Dataset): The training dataset.
        collate_fn (function): Function to collate data samples into batch tensors.
        loss_fn (function): Loss function to be used during training.

    Returns:
        float: The average loss over the training epoch.
    """
    model.train()

    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    losses = 0
    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        generator=torch.Generator(device='cuda')
    )
    total = math.ceil(len(train_dataloader.dataset) / train_dataloader.batch_size)

    for i, (src, tgt) in tqdm(enumerate(train_dataloader), total=total, dynamic_ncols=True):
        # Enable autocasting for mixed precision training
        try:
            with torch.autocast(device_type=str(DEVICE), dtype=torch.bfloat16, enabled=True):
                src = src.to(DEVICE) 
                tgt = tgt.to(DEVICE)

                tgt_input = tgt[:-1, :]  # (T, B)

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                    src, tgt_input)
                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                                tgt_padding_mask, src_padding_mask)  # (T, B, tgt_vocab_size)
                assert logits.dtype is torch.bfloat16
                tgt_out = tgt[1:, :].type(torch.long)  # (T, B)
                loss = loss_fn(
                    logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                loss = loss/grad_accum_steps

            scaler.scale(loss).backward()
            # if (i+1)% grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            # loss.backward()
            scaler.step(optimizer)
            # optimizer.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses += loss.item()
        except:
            with open('training.log', 'a') as log:
                log.write(str(logits.size()))
                log.write('\n')
            continue

        if i % 500 == 0:
            wandb.log({"train_loss:": loss.item()})
        # Explicitly delete variables to free up memory
        del src, tgt, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, logits, tgt_out, loss
        torch.cuda.empty_cache()
        gc.collect()
    return losses / len(list(train_dataloader))

# Certified


def train(num_epochs, train_data, val_data, en_test, vi_test, collate_fn, text_transform, vocab_transform, VOCAB_SIZE):
    """
    Trains the model and evaluates it over multiple epochs, logging the performance
    and saving the best model based on BLEU score.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        num_epochs (int): Number of epochs to train the model.
        train_data (Dataset): The training dataset.
        val_data (Dataset): The validation dataset.
        en_test (list): The English test dataset.
        vi_test (list): The Vietnamese test dataset.
        collate_fn (function): Function to collate data samples into batch tensors.
        loss_fn (function): Loss function to be used during training.
        text_transform (function): Function to transform text to IDs.
        vocab_transform (Vocab): Vocabulary object for converting IDs back to words.
    """
    transformer = Seq2SeqTransformer(cfg.num_encoder_layers, cfg.num_decoder_layers, cfg.emb_size,
                                     cfg.nhead, VOCAB_SIZE, VOCAB_SIZE, cfg.ffn_hid_dim, cfg.dropout, cfg.activation)
    optimizer = torch.optim.AdamW(params=transformer.parameters(
    ), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, label_smoothing=0.1).cuda()
    
    if cfg.load_checkpoint:
        checkpoint = torch.load(cfg.checkpoint_path)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    transformer = transformer.to(DEVICE)
    if cfg.train_from_scratch:
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    # Calculation the total parameters and trainable parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    total_trainable_params = sum(
        p.numel() for p in transformer.parameters() if p.requires_grad)
    print(
        f"Total parameters: {total_params / 1e6:.2f} M, Trainable parameters: {total_trainable_params / 1e6:.2f} M")
    
    # Log hyperparameters
    with open('training.log', 'a') as log:
        t = datetime.now(tz=timezone('Asia/Ho_Chi_Minh'))
        log.write(
            f"[{t.month:0>2}/{t.day:0>2}/{t.year} - {t.hour:0>2}:{t.minute:0>2}:{t.second:0>2}] IWSLT - Start new training session!")
        log.write(f"\nvocab size: {VOCAB_SIZE}")
        log.write(
            f"\nTotal parameters: {total_params / 1e6:.2f} M, Trainable parameters: {total_trainable_params / 1e6:.2f} M\n")
        log.write(f"{cfg.show_config()}")
    
    patient = 10
    best_model_weight = None
    best_bleu_score = 0.0
    bleu_score_greedy = 0.0

    print("First")
    """
    Start training
    """
    wandb.init(
        project='EnViT5_tokenize',
        config={
            "learning_rate": cfg.learning_rate,
            "architecture": "Transformer",
            "dataset": "IWSLT",
            "epochs": cfg.num_epochs,
        }
    )

    try:
        for epoch in range(1, num_epochs+1):
            try:
                train_loss = train_epoch(
                    model=transformer, optimizer=optimizer, train_data=train_data, collate_fn=collate_fn, loss_fn=loss_fn)

            except KeyboardInterrupt:
                print('Interrupted')
                try:
                    sys.exit(130)
                except SystemExit:
                    os._exit(130)

            val_loss = evaluate_model(
                transformer, val_data, cfg.batch_size, collate_fn, loss_fn)
            # if epoch % 5 == 0:
            bleu_score_greedy = calculate_bleu_greedy(
                transformer, en_test, vi_test, text_transform, vocab_transform)
            print("greedy:", bleu_score_greedy)
            bleu_score_greedy = float(str(bleu_score_greedy)[6:12])

            bleu_score = bleu_score_greedy
            print(
                f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
            print(f"BLEU Score of current model: {bleu_score}")

            wandb.log({
                "val_loss:": val_loss,
                "bleu_score_greedy": bleu_score_greedy,
                "max_bleu_score": bleu_score,
            })

            if bleu_score > best_bleu_score:
                best_bleu_score = bleu_score
                best_model_weight = copy.deepcopy(transformer.state_dict())
                best_optimizer_weight = copy.deepcopy(optimizer.state_dict())
                torch.save({
                    "model_state_dict": transformer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, cfg.save_model_path)
                print("Model completed! Saved at", cfg.save_model_path)
                patient = 10

            else:
                patient -= 1
                print("Patient reduced to", patient)
                if patient == 0:
                    print("Early stopping due to increasing BLEU score.")
                    break
            t = datetime.now()
            with open(f"{cfg.folder_path}/training.log", "a") as log:
                print("Writing to log...")
                t = datetime.now(tz=timezone('Asia/Ho_Chi_Minh'))
                log.write(f"[{t.month:0>2}/{t.day:0>2}/{t.year} - {t.hour:0>2}:{t.minute:0>2}:{t.second:0>2}] Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, BLEU: {bleu_score}\n\n")

            torch.cuda.empty_cache()
            gc.collect()
    except KeyboardInterrupt:
        print("Canceled by user.")

    transformer.load_state_dict(best_model_weight)
    optimizer.load_state_dict(best_optimizer_weight)

    print("Saving model...")
    # save_checkpoint(transformer, hyperparameters=hyperparameters, path=PATH)
    torch.save({
        "model_state_dict": transformer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, cfg.save_model_path)
    print("Model completed! Saved at", cfg.save_model_path)
    print("Final loss:", evaluate_model(transformer,
            val_data, cfg.batch_size, collate_fn, loss_fn))
    torch.cuda.empty_cache()
    gc.collect()

def generate_new_data(transformer, text_transform, vocab_transform):
    _, en_train, _, vi_train, _ = read_dataset(cfg.data_name)
    en_translate = []
    vi_translate = []
    for en in en_train:
        vi_translate.append(translate_greedy(transformer, en, text_transform, vocab_transform))
    for vi in vi_train:
        en_translate.append(translate_greedy(transformer, vi, text_transform, vocab_transform))
    src_train = en_translate + vi_translate
    tgt_train = vi_train + en_train
    src_train, tgt_train = shuffle_data(src_train, tgt_train)
    src_train_final, val_src, tgt_train_final, val_tgt = train_test_split(src_train, tgt_train, test_size=1000, shuffle=True)
    train_data = MTDataset(src_train_final, tgt_train_final, split='train')
    val_data = MTDataset(val_src, val_tgt, split='val')
    return train_data, val_data

def train_multi_phrase(num_epochs, num_phrases, train_data, val_data, en_test, vi_test, collate_fn, text_transform, vocab_transform, VOCAB_SIZE):
    """
    Trains the model and evaluates it over multiple epochs, logging the performance
    and saving the best model based on BLEU score.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        num_epochs (int): Number of epochs to train the model.
        train_data (Dataset): The training dataset.
        val_data (Dataset): The validation dataset.
        en_test (list): The English test dataset.
        vi_test (list): The Vietnamese test dataset.
        collate_fn (function): Function to collate data samples into batch tensors.
        loss_fn (function): Loss function to be used during training.
        text_transform (function): Function to transform text to IDs.
        vocab_transform (Vocab): Vocabulary object for converting IDs back to words.
    """
    transformer = Seq2SeqTransformer(cfg.num_encoder_layers, cfg.num_decoder_layers, cfg.emb_size,
                                     cfg.nhead, VOCAB_SIZE, VOCAB_SIZE, cfg.ffn_hid_dim, cfg.dropout, cfg.activation)
    optimizer = torch.optim.AdamW(params=transformer.parameters(
    ), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, label_smoothing=0.1).cuda()
    
    best_bleu_score = 0.0
    bleu_score_greedy = 0.0
    if cfg.load_checkpoint:
        checkpoint = torch.load(cfg.checkpoint_path)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        bleu_score_greedy = calculate_bleu_greedy(transformer, en_test, vi_test, text_transform, vocab_transform)
        print("First BLEU score:")
        print("bleu_score_greedy:", bleu_score_greedy)
        bleu_score_greedy = float(str(bleu_score_greedy)[6:12])

    transformer = transformer.to(DEVICE)
    if cfg.train_from_scratch:
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    # Calculation the total parameters and trainable parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    total_trainable_params = sum(
        p.numel() for p in transformer.parameters() if p.requires_grad)
    print(
        f"Total parameters: {total_params / 1e6:.2f} M, Trainable parameters: {total_trainable_params / 1e6:.2f} M")
    
    # Log hyperparameters
    with open(f"{cfg.folder_path}/training_multi_phrases.log", 'a') as log:
        t = datetime.now(tz=timezone('Asia/Ho_Chi_Minh'))
        log.write(
            f"[{t.month:0>2}/{t.day:0>2}/{t.year} - {t.hour:0>2}:{t.minute:0>2}:{t.second:0>2}] IWSLT - Start new training session!")
        log.write(f"\nvocab size: {VOCAB_SIZE}")
        log.write(
            f"\nTotal parameters: {total_params / 1e6:.2f} M, Trainable parameters: {total_trainable_params / 1e6:.2f} M\n")
        log.write(f"{cfg.get_config()}")
    
    best_model_weight = None
    best_bleu_score = bleu_score_greedy

    print("First")
    """
    Start training
    """
    wandb.init(
        project='EnViT5_tokenize',
        config={
            "learning_rate": cfg.learning_rate,
            "architecture": "Transformer",
            "dataset": "IWSLT",
            "epochs": cfg.num_epochs,
        }
    )
    for phrase in range(num_phrases):
        patient = 5
        if cfg.load_checkpoint:
            train_data, val_data = generate_new_data(transformer, text_transform, vocab_transform)
        try:
            for epoch in range(1, num_epochs+1):
                try:
                    train_loss = train_epoch(
                        model=transformer, optimizer=optimizer, train_data=train_data, collate_fn=collate_fn, loss_fn=loss_fn)

                except KeyboardInterrupt:
                    print('Interrupted')
                    try:
                        sys.exit(130)
                    except SystemExit:
                        os._exit(130)

                val_loss = evaluate_model(
                    transformer, val_data, cfg.batch_size, collate_fn, loss_fn)
                bleu_score_greedy = calculate_bleu_greedy(
                    transformer, en_test, vi_test, text_transform, vocab_transform)
                print("greedy:", bleu_score_greedy)
                bleu_score_greedy = float(str(bleu_score_greedy)[6:12])

                bleu_score = bleu_score_greedy
                print(
                    f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
                print(f"\nBLEU Score of current model: {bleu_score}")

                wandb.log({
                    f"val_loss:": val_loss,
                    "bleu_score_greedy": bleu_score_greedy,
                    "max_bleu_score": bleu_score,
                })

                if bleu_score > best_bleu_score:
                    best_bleu_score = bleu_score
                    best_model_weight = copy.deepcopy(transformer.state_dict())
                    best_optimizer_weight = copy.deepcopy(optimizer.state_dict())
                    torch.save({
                        "model_state_dict": transformer.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, cfg.save_model_path)
                    print("Model completed! Saved at", cfg.save_model_path)
                    patient = 5

                else:
                    patient -= 1
                    print("Patient reduced to", patient)
                    if patient == 0:
                        # train_data, val_data = generate_new_data(transformer, text_transform, vocab_transform)
                        print("Early stopping due to increasing BLEU score.")
                        break
                t = datetime.now()
                with open(f"{cfg.folder_path}/training_multi_phrases.log", "a") as log:
                    print("Writing to log...")
                    t = datetime.now(tz=timezone('Asia/Ho_Chi_Minh'))
                    log.write(f"[{t.month:0>2}/{t.day:0>2}/{t.year} - {t.hour:0>2}:{t.minute:0>2}:{t.second:0>2}] Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, BLEU: {bleu_score}\n\n")

                torch.cuda.empty_cache()
                gc.collect()
        except KeyboardInterrupt:
            print("Canceled by user.")

    transformer.load_state_dict(best_model_weight)
    optimizer.load_state_dict(best_optimizer_weight)

    print("Saving model...")
    # save_checkpoint(transformer, hyperparameters=hyperparameters, path=PATH)
    torch.save({
        "model_state_dict": transformer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, cfg.save_model_path)
    print("Model completed! Saved at", cfg.save_model_path)
    print("Final loss:", evaluate_model(transformer,
            val_data, cfg.batch_size, collate_fn, loss_fn))
    torch.cuda.empty_cache()
    gc.collect()