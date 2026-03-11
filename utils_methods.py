from utils_libs import *


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_batch_grammar(input_batch,
                            target_batch,
                            c_mask_batch,
                            model,
                            device,
                            V_ATC_ids,
                            lambda_vocab: float = 0.1):
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    c_mask_batch = c_mask_batch.to(device)
    V_ATC_ids    = V_ATC_ids.to(device)

    logits = model(input_batch)                # [B, T, V]
    B, T, V = logits.shape

    # 1) Standard CLM term
    clm_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )

    # 2) Grammar-informed vocab term
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)          # [B, T, V]
    log_probs_ATC = log_probs.index_select(dim=-1, index=V_ATC_ids)
    log_P_ATC = torch.logsumexp(log_probs_ATC, dim=-1) # [B, T]

    constrained_count = c_mask_batch.sum()
    if constrained_count > 0:
        vocab_loss = -(c_mask_batch * log_P_ATC).sum() / constrained_count
    else:
        vocab_loss = torch.tensor(0.0, device=device)

    total_loss = clm_loss + lambda_vocab * vocab_loss
    return total_loss


'''def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss'''


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def calc_loss_loader_grammar(data_loader, model, device, V_ATC_ids=None, lambda_vocab=0.0, num_batches=None):
    """
    Compute average loss over loader.
    If V_ATC_ids is provided, uses grammar-informed loss.
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    
    num_batches = min(num_batches or len(data_loader), len(data_loader))
    
    for i, batch_data in enumerate(data_loader):
        if i >= num_batches:
            break
            
        if V_ATC_ids is not None and len(batch_data) == 3:
            # Grammar mode: (inputs, targets, c_mask)
            input_batch, target_batch, c_mask_batch = [x.to(device) for x in batch_data]
            loss = calc_loss_batch_grammar(  # your grammar loss fn
                input_batch, target_batch, c_mask_batch, 
                model, device, V_ATC_ids, lambda_vocab
            )
        else:
            # Pure CLM mode: (inputs, targets)
            input_batch, target_batch = [x.to(device) for x in batch_data[:2]]
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
        total_loss += loss.item()
    
    return total_loss / num_batches




def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def evaluate_model_grammar(model, train_loader, val_loader, device, eval_iter, V_ATC_ids=None, lambda_vocab=0.0):
    model.eval()
    with torch.no_grad():
        #train_loss = calc_loss_loader_grammar(train_loader, model, device, num_batches=eval_iter)
        #val_loss = calc_loss_loader_grammar(val_loader, model, device, num_batches=eval_iter)

        train_loss = calc_loss_loader_grammar(train_loader, model, device, V_ATC_ids=V_ATC_ids, lambda_vocab=lambda_vocab, num_batches=eval_iter)
        val_loss = calc_loss_loader_grammar(val_loader, model, device, V_ATC_ids=V_ATC_ids, lambda_vocab=lambda_vocab, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss


def generate_text_simple(model, idx, max_new_tokens, context_size, repetition_penalty=1.2):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # --- REPETITION PENALTY LOGIC ---
        for i in range(logits.shape[0]):  # Batch loop
            # Find unique tokens already in the sequence for this batch item
            for token_id in set(idx[i].tolist()):
                # If logit is positive, divide to reduce it; if negative, multiply to make more negative
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= repetition_penalty
                else:
                    logits[i, token_id] *= repetition_penalty
        # --------------------------------

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx



def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()

    # Safely check if pos_emb exists; default to None if it doesn't
    pos_emb = getattr(model, "pos_emb", None)

    # Use config instead of non-existent pos_emb
    if pos_emb is None:
        context_size = model.cfg["context_length"] 
    else:
        context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


# PRETRAINNING FOR NEXT TOKEN GENERATION

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen



# Train Grammar informed Model
def train_model_simple_with_grammar(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       use_grammar_loss=True, V_ATC_ids=None, lambda_vocab=0.1):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            if use_grammar_loss:
                input_batch, target_batch, c_mask_batch = batch
                loss = calc_loss_batch_grammar(
                    input_batch, target_batch, c_mask_batch,
                    model, device, V_ATC_ids, lambda_vocab=lambda_vocab
                )
            else:
                input_batch, target_batch = batch
                loss = calc_loss_batch(
                    input_batch, target_batch, model, device
                )

            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1
            # ... evaluation + logging unchanged ...
            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_grammar(
                    model, train_loader, val_loader, device, eval_iter, V_ATC_ids=V_ATC_ids, lambda_vocab=lambda_vocab)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, model_size):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Add the title
    ax1.set_title(f"Training and Validation Loss for Model {model_size}")

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plot_name = f"loss-plot-standalone-{model_size}.pdf"
    print(f"Plot saved as {plot_name}")
    plt.savefig(plot_name)
    # plt.show()


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples



def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


# GENERATE USING OPEN WEIGHTS FROM GPT-2/ or other MODELS like LLAMA
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None, repetition_penalty=1.2):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # --- REPETITION PENALTY LOGIC ---
        for i in range(logits.shape[0]):  # Batch loop
            # Find unique tokens already in the sequence for this batch item
            for token_id in set(idx[i].tolist()):
                # If logit is positive, divide to reduce it; if negative, multiply to make more negative
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= repetition_penalty
                else:
                    logits[i, token_id] *= repetition_penalty
        # --------------------------------


        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

