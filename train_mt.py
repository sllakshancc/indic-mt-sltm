import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    EncoderDecoderModel,
    EncoderDecoderConfig,
    BertConfig
)
from tqdm import tqdm

# =========================
# CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 1.0                 # base LR (scaled by scheduler)
WARMUP_STEPS = 4000
LABEL_SMOOTHING = 0.1

VOCAB_SIZE_SI = 8000     # change
VOCAB_SIZE_TA = 8000     # change

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================
# DATASET
# =========================

class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer_src, tokenizer_tgt):
        self.pairs = pairs
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        src_ids = self.tokenizer_src.encode(src)[:MAX_LEN-2]
        tgt_ids = self.tokenizer_tgt.encode(tgt)[:MAX_LEN-2]

        src_ids = [BOS_ID] + src_ids + [EOS_ID]
        tgt_ids = [BOS_ID] + tgt_ids + [EOS_ID]

        return {
            "input_ids": torch.tensor(src_ids),
            "labels": torch.tensor(tgt_ids)
        }


def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]

    input_ids = nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=PAD_ID
    )
    labels = nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=PAD_ID
    )

    return {"input_ids": input_ids, "labels": labels}


# =========================
# MODEL
# =========================

def build_model():

    encoder_config = BertConfig(
        vocab_size=VOCAB_SIZE_SI,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        pad_token_id=PAD_ID,
    )

    decoder_config = BertConfig(
        vocab_size=VOCAB_SIZE_TA,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        is_decoder=True,
        add_cross_attention=True,
        pad_token_id=PAD_ID,
    )

    config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )

    config.decoder_start_token_id = BOS_ID
    config.eos_token_id = EOS_ID
    config.pad_token_id = PAD_ID

    return EncoderDecoderModel(config)


# =========================
# TRANSFORMER LR SCHEDULER
# =========================

class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


# =========================
# LABEL SMOOTHING LOSS
# =========================

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing, vocab_size, ignore_index=PAD_ID):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(
                -1,
                target.unsqueeze(-1),
                1.0 - self.smoothing
            )
            true_dist[target == self.ignore_index] = 0

        loss = -torch.sum(true_dist * log_probs, dim=-1)
        mask = target != self.ignore_index
        return loss[mask].mean()


# =========================
# TRAIN / EVAL
# =========================

def train_epoch(model, loader, optimizer, scheduler, scaler, criterion):

    model.train()
    total_loss = 0

    for batch in tqdm(loader):

        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=labels[:, :-1]
            )
            logits = outputs.logits
            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE_TA),
                labels[:, 1:].reshape(-1)
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):

    model.eval()
    total_loss = 0

    for batch in loader:

        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=labels[:, :-1]
        )
        logits = outputs.logits

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE_TA),
            labels[:, 1:].reshape(-1)
        )

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# CHECKPOINT
# =========================

def save_checkpoint(model, optimizer, scheduler, epoch, best_val):

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_step": scheduler.step_num,
        "epoch": epoch,
        "best_val": best_val
    }, os.path.join(CHECKPOINT_DIR, "latest.pt"))


def load_checkpoint(model, optimizer, scheduler):

    path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if not os.path.exists(path):
        return 0, float("inf")

    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.step_num = ckpt["scheduler_step"]

    print("Resumed from checkpoint")
    return ckpt["epoch"], ckpt["best_val"]


# =========================
# MAIN
# =========================

def main():

    dataset = load_dataset("opus100", "si-ta")

    train_pairs = [(x["translation"]["si"], x["translation"]["ta"])
                   for x in dataset["train"]]

    val_pairs = [(x["translation"]["si"], x["translation"]["ta"])
                 for x in dataset["validation"]]

    train_loader = DataLoader(
        TranslationDataset(train_pairs, tokenizer_si, tokenizer_ta),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        TranslationDataset(val_pairs, tokenizer_si, tokenizer_ta),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = build_model().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = TransformerLRScheduler(
        optimizer,
        d_model=512,
        warmup_steps=WARMUP_STEPS
    )

    scaler = torch.cuda.amp.GradScaler()
    criterion = LabelSmoothingLoss(
        LABEL_SMOOTHING,
        VOCAB_SIZE_TA
    )

    start_epoch, best_val = load_checkpoint(
        model, optimizer, scheduler
    )

    for epoch in range(start_epoch, EPOCHS):

        print(f"\nEpoch {epoch+1}")

        train_loss = train_epoch(
            model, train_loader,
            optimizer, scheduler,
            scaler, criterion
        )

        val_loss = evaluate(
            model, val_loader, criterion
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "best_model.pt")
            )
            print("Saved best model")

        save_checkpoint(
            model, optimizer, scheduler,
            epoch + 1, best_val
        )


if __name__ == "__main__":
    main()