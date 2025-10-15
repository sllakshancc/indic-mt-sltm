import os
import re
import time
import pickle
import regex
import grapheme
import torch
import json
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    EncoderDecoderModel,
    BertConfig,
    EncoderDecoderConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import sacrebleu


class GPETokenizer:
    """GPE Tokenizer with HuggingFace-compatible interface"""

    def __init__(self, vocab_size=16000, dummy_prefix="▁"):
        self.vocab_size = vocab_size
        self.dummy_prefix = dummy_prefix

        self.vocab = {}       # id -> string
        self.vocab_re = {}    # string -> id
        self.merges = {}      # (id1, id2) -> new_id
        self.trained = False

        # Special tokens
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3
        }

        # Regex pattern for text chunking
        #self.whitespace_pattern = r"\w+|[^\w\s]"
        self.whitespace_pattern = r" ?\w+| ?[^\w\s]+"

        # HF compatibility attributes
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def train(self, texts):
        """Train tokenizer on list of text strings."""
        print(f"Training GPE tokenizer on {len(texts)} texts...")

        # Initialize vocab with special tokens
        self.vocab = {v: k for k, v in self.special_tokens.items()}
        self.vocab_re = self.special_tokens.copy()

        # Preprocess texts
        if self.dummy_prefix:
            texts = [self.dummy_prefix + regex.sub(r"\s+", " ", t.strip()) for t in texts]
        else:
            texts = [regex.sub(r"\s+", " ", t.strip()) for t in texts]

        # Collect initial graphemes
        initial_graphemes = set()
        for text in tqdm(texts[:10000], desc="Collecting graphemes"):  # Limit for speed
            text_chunks = regex.findall(self.whitespace_pattern, text)
            text_chunks = [t.replace(" ", "▁") for t in text_chunks if t.strip()]

            for chunk in text_chunks:
                graphemes_list = list(grapheme.graphemes(chunk))
                initial_graphemes.update(graphemes_list)

        # Add graphemes to vocab
        current_id = len(self.vocab)
        for g in sorted(initial_graphemes):
            if g not in self.vocab_re:
                self.vocab[current_id] = g
                self.vocab_re[g] = current_id
                current_id += 1

        print(f"Initial vocab size: {len(self.vocab)}")

        # Calculate number of merges needed
        num_merges = min(self.vocab_size - len(self.vocab), 5000)  # Cap merges for speed

        # Convert texts to IDs for merging
        ids_list = self._convert_to_ids_train(texts[:10000])  # Limit for speed

        # Perform merges
        for i in tqdm(range(num_merges), desc="Merging"):
            stats = {}
            for chunk_ids in ids_list:
                self._get_stats(chunk_ids, stats)

            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = len(self.vocab)

            # Update IDs with merge
            ids_list = [self._merge(chunk_ids, pair, idx) for chunk_ids in ids_list]

            # Record merge
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if i % 100 == 0:
                print(f"Merge {i}: {self.vocab[pair[0]]} + {self.vocab[pair[1]]} -> {self.vocab[idx]}")

        # Update reverse vocab
        self.vocab_re = {v: k for k, v in self.vocab.items()}

        print(f"Training complete. Final vocab size: {len(self.vocab)}")
        self.trained = True

    def encode(self, text, add_special_tokens=True, max_length=None, truncation=False, padding=False):
        """Encode text to token IDs."""
        if not self.trained:
            raise ValueError("Tokenizer not trained")

        # Handle batch input
        if isinstance(text, list):
            return [self.encode(t, add_special_tokens, max_length, truncation, padding) for t in text]

        # Process text
        text_chunks = regex.findall(self.whitespace_pattern, text)
        text_chunks = [t.replace(" ", "▁") for t in text_chunks if t.strip()]

        ids = []
        for chunk in text_chunks:
            graphemes_list = list(grapheme.graphemes(chunk))
            for g in graphemes_list:
                if g in self.vocab_re:
                    ids.append(self.vocab_re[g])
                else:
                    ids.append(self.unk_token_id)


        # Add special tokens
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        # Truncation
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]

        # Padding
        if padding and max_length:
            if len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))

        return ids

    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs to text."""
        if isinstance(ids[0], list):
            return [self.decode(i, skip_special_tokens) for i in ids]

        tokens = []

        text = "".join(tokens).replace("▁", " ")
        return text.strip()



    def tokenize(self, text):
        """Tokenize text into subword strings."""
        ids = self.encode(text, add_special_tokens=False)
        return [self.vocab.get(i, self.unk_token) for i in ids]

    def __call__(self, text, **kwargs):
        """HuggingFace-compatible call interface."""
        if isinstance(text, str):
            text = [text]

        max_length = kwargs.get('max_length', None)
        padding = kwargs.get('padding', False)
        truncation = kwargs.get('truncation', False)
        return_tensors = kwargs.get('return_tensors', None)

        if padding == "max_length":
            padding = True

        encoded = []
        attention_masks = []

        for t in text:
            ids = self.encode(t, max_length=max_length, truncation=truncation, padding=padding)
            encoded.append(ids)

            # Create attention mask
            mask = [1 if i != self.pad_token_id else 0 for i in ids]
            attention_masks.append(mask)

        result = {
            'input_ids': encoded,
            'attention_mask': attention_masks
        }

        if return_tensors == "pt":
            result['input_ids'] = torch.tensor(result['input_ids'])
            result['attention_mask'] = torch.tensor(result['attention_mask'])

        return result

    def save(self, path):
        """Save tokenizer to disk."""
        os.makedirs(path, exist_ok=True)

        save_dict = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }

        with open(os.path.join(path, 'gpe_tokenizer.pkl'), 'wb') as f:
            pickle.dump(save_dict, f)

        # Save config for HF compatibility
        config = {
            'tokenizer_class': 'GPETokenizerHF',
            'vocab_size': len(self.vocab),
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token
        }

        with open(os.path.join(path, 'tokenizer_config.json'), 'w') as f:
            json.dump(config, f)

    def load(self, path):
        """Load tokenizer from disk."""
        with open(os.path.join(path, 'gpe_tokenizer.pkl'), 'rb') as f:
            save_dict = pickle.load(f)

        self.vocab = save_dict['vocab']
        self.merges = save_dict['merges']
        self.special_tokens = save_dict['special_tokens']
        self.vocab_size = save_dict['vocab_size']

        self.vocab_re = {v: k for k, v in self.vocab.items()}
        self.trained = True

    # Helper methods
    def _get_stats(self, ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def _apply_merges(self, ids):
        """Apply learned merges to a sequence of IDs."""
        made_merge = True
        while made_merge:
            made_merge = False
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) in self.merges:
                    new_ids.append(self.merges[(ids[i], ids[i+1])])
                    i += 2
                    made_merge = True
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def _convert_to_ids_train(self, texts):
        """Convert texts to IDs for training."""
        ids_list = []
        for text in texts:
            text_chunks = regex.findall(self.whitespace_pattern, text)
            text_chunks = [t.replace(" ", "▁") for t in text_chunks if t.strip()]

            for chunk in text_chunks:
                graphemes_list = list(grapheme.graphemes(chunk))
                chunk_ids = []
                for g in graphemes_list:
                    if g in self.vocab_re:
                        chunk_ids.append(self.vocab_re[g])
                if chunk_ids:
                    ids_list.append(chunk_ids)

        return ids_list

    def pad(self, encoded_inputs, padding=True, max_length=None, return_tensors=None, **kwargs):
      """
      HuggingFace-compatible pad method.
      Can handle a single dict or a list of dicts with 'input_ids' and 'attention_mask'.
      """
      # Case 1: already a batch dict
      if isinstance(encoded_inputs, dict) and "input_ids" in encoded_inputs:
          input_ids = encoded_inputs["input_ids"]
          attention_mask = encoded_inputs.get("attention_mask", None)

      # Case 2: list of dicts (from DataCollator)
      elif isinstance(encoded_inputs, (list, tuple)):
          input_ids = [e["input_ids"] for e in encoded_inputs]
          attention_mask = [e.get("attention_mask") for e in encoded_inputs]
          if all(am is None for am in attention_mask):
              attention_mask = None
      else:
          raise TypeError("Unsupported input format for pad()")

      # Convert tensors to lists if needed
      if isinstance(input_ids, torch.Tensor):
          input_ids = input_ids.tolist()
      if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
          attention_mask = attention_mask.tolist()

      max_len = max_length or max(len(seq) for seq in input_ids)

      padded_ids, padded_mask = [], []
      for i, seq in enumerate(input_ids):
          # Ensure list, not tensor
          if isinstance(seq, torch.Tensor):
              seq = seq.tolist()

          pad_len = max_len - len(seq)
          padded_seq = seq + [self.pad_token_id] * pad_len
          padded_ids.append(padded_seq)

          if attention_mask is not None and attention_mask[i] is not None:
              mask_seq = attention_mask[i]
              if isinstance(mask_seq, torch.Tensor):
                  mask_seq = mask_seq.tolist()
              mask_seq = mask_seq + [0] * pad_len
          else:
              mask_seq = [1] * len(seq) + [0] * pad_len
          padded_mask.append(mask_seq)

      result = {
          "input_ids": torch.tensor(padded_ids) if return_tensors == "pt" else padded_ids,
          "attention_mask": torch.tensor(padded_mask) if return_tensors == "pt" else padded_mask,
      }
      return result



# Load dataset
ds = load_dataset("Helsinki-NLP/opus-100", "en-si")
print(f"Dataset structure: {ds}")
print(f"Sample translation: {ds['train']['translation'][0]}")

# Extract texts for training
train_texts_en = [ex["en"] for ex in ds["train"]["translation"]]
train_texts_si = [ex["si"] for ex in ds["train"]["translation"]]

val_texts_en = [ex["en"] for ex in ds["validation"]["translation"]]
val_texts_si = [ex["si"] for ex in ds["validation"]["translation"]]

print(f"Training samples: {len(train_texts_en)}")
print(f"Validation samples: {len(val_texts_en)}")



# Initialize tokenizers
tokenizer_en = GPETokenizerHF(vocab_size=16000)
tokenizer_si = GPETokenizerHF(vocab_size=16000)

# Train English tokenizer
print("Training English GPE tokenizer...")
tokenizer_en.train(train_texts_en)

# Train Sinhala tokenizer
print("\nTraining Sinhala GPE tokenizer...")
tokenizer_si.train(train_texts_si)

# Save tokenizers
tokenizer_en.save("gpe_tokenizer_en")
tokenizer_si.save("gpe_tokenizer_si")

print("\nTokenizers saved successfully!")



# Initialize tokenizers
tokenizer_en = GPETokenizerHF(vocab_size=16000)
tokenizer_si = GPETokenizerHF(vocab_size=16000)

tokenizer_en.load("en")
tokenizer_si.load("si")


# Test tokenization on sample sentences
num_samples = 5
samples = ds["validation"].select(range(num_samples))

for i, ex in enumerate(samples["translation"]):
    en_sent = ex["en"]
    si_sent = ex["si"]

    # Tokenize
    en_tokens = tokenizer_en.tokenize(en_sent)
    en_ids = tokenizer_en.encode(en_sent)

    si_tokens = tokenizer_si.tokenize(si_sent)
    si_ids = tokenizer_si.encode(si_sent)

    print(f"\nSample {i+1}")
    print(f"EN: {en_sent}")
    print(f"EN tokens (first 10): {en_tokens}")
    print(f"EN IDs (first 10): {en_ids}")

    print(f"SI: {si_sent}")
    print(f"SI tokens (first 10): {si_tokens}")
    print(f"SI IDs (first 10): {si_ids}")
    print("-" * 60)

# Test encode-decode roundtrip
for i in range(5):
    ref_si = ds["validation"][i]["translation"]["si"]

    encoded = tokenizer_si.encode(ref_si)
    decoded = tokenizer_si.decode(encoded, skip_special_tokens=True)

    print(f"\nOriginal: {ref_si}")
    print(f"Decoded:  {decoded}")


max_len = 64

def encode_batch(batch):
    """Encode a batch of translations."""
    en_texts = [item["en"] for item in batch]
    si_texts = [item["si"] for item in batch]

    # Use GPE tokenizers
    en = tokenizer_en(en_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    si = tokenizer_si(si_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")

    return {
        "input_ids": en["input_ids"],
        "attention_mask": en["attention_mask"],
        "labels": si["input_ids"]
    }

# Process datasets
print("Encoding training dataset...")
train_ds = ds["train"].map(
    lambda x: encode_batch(x["translation"]),
    batched=True,
    remove_columns=["translation"],
    batch_size=32
)

print("Encoding validation dataset...")
val_ds = ds["validation"].map(
    lambda x: encode_batch(x["translation"]),
    batched=True,
    remove_columns=["translation"],
    batch_size=32
)

# Set format for PyTorch
train_ds.set_format(type="torch")
val_ds.set_format(type="torch")

print(f"Encoded train samples: {len(train_ds)}")
print(f"Encoded val samples: {len(val_ds)}")

batch = [ {"en": "Hello world!", "si": "හෙලෝ ලෝකය!"} ]
encoded = encode_batch(batch)
print({k: v.shape for k, v in encoded.items()})

print(train_ds[0]['input_ids'].shape)  # should be torch.Size([64])
print(train_ds[0]['labels'].shape)     # should be torch.Size([64])

# Check token ranges (not all 0s or 1s)
print(train_ds[0]['input_ids'][:10])
print(train_ds[0]['labels'][:10])


from transformers import EncoderDecoderModel, BertConfig, BertModel, EncoderDecoderConfig

# Encoder config (English)
encoder_config = BertConfig(
    vocab_size=len(tokenizer_en),
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    max_position_embeddings=128,
    pad_token_id=tokenizer_en.pad_token_id,
    bos_token_id=tokenizer_en.bos_token_id,
    eos_token_id=tokenizer_en.eos_token_id,
)

# Decoder config (Sinhala) – note: add_is_decoder & cross-attention
decoder_config = BertConfig(
    vocab_size=len(tokenizer_si),
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    max_position_embeddings=128,
    pad_token_id=tokenizer_si.pad_token_id,
    bos_token_id=tokenizer_si.bos_token_id,
    eos_token_id=tokenizer_si.eos_token_id,
    is_decoder=True,
    add_cross_attention=True,
)

#model = EncoderDecoderModel.from_encoder_decoder_config(
#    encoder_config=encoder_config,
#    decoder_config=decoder_config
#)

# Build encoder-decoder model
#encoder_model = BertModel(config=encoder_config)
#decoder_model = BertModel(config=decoder_config)

#model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model)
#model = EncoderDecoderModel(encoder=encoder_config, decoder=decoder_config)
config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
model = EncoderDecoderModel(config=config)

# Tie embeddings to reduce params (optional but helps)
model.config.decoder_start_token_id = tokenizer_si.bos_token_id
model.config.pad_token_id = tokenizer_si.pad_token_id


# Print model config (layers, heads, hidden sizes, etc.)
print(model)


from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

batch_size = 16

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    logging_steps=500,
    save_steps=1000,
    eval_steps=1000,
    num_train_epochs=1,  # for demo, increase for real
    fp16=torch.cuda.is_available(),
    report_to=[],
)


from dataclasses import dataclass
from typing import Any, Dict, List
import torch

@dataclass
class CustomDataCollator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack and pad manually (your data is already padded, so this is simple)
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])

        # Replace padding in labels with -100 to ignore loss there
        labels[labels == self.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

data_collator = CustomDataCollator(pad_token_id=tokenizer_si.pad_token_id)


import sacrebleu

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = [tokenizer_si.decode(p, skip_special_tokens=True) for p in preds]
    decoded_labels = [tokenizer_si.decode(l, skip_special_tokens=True) for l in labels]

    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels], force=True)
    return {"bleu": bleu.score}


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=None,   # ✅ don’t pass custom tokenizer
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


trainer.train()
metrics = trainer.evaluate()
print(metrics)


import matplotlib.pyplot as plt

# After training
history = trainer.state.log_history

steps = []
train_loss = []
eval_loss = []
bleu = []

for record in history:
    if "loss" in record.keys() and "step" in record.keys():
        steps.append(record["step"])
        train_loss.append(record["loss"])
    if "eval_loss" in record.keys():
        eval_loss.append(record["eval_loss"])
    if "eval_bleu" in record.keys():
        bleu.append(record["eval_bleu"])

plt.figure(figsize=(10,5))
plt.plot(steps[:len(train_loss)], train_loss, label="Train Loss")
if eval_loss:
    plt.plot(steps[:len(eval_loss)], eval_loss, label="Validation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.show()

if bleu:
    plt.figure(figsize=(10,5))
    plt.plot(steps[:len(bleu)], bleu, label="Validation BLEU")
    plt.xlabel("Steps")
    plt.ylabel("BLEU Score")
    plt.legend()
    plt.title("BLEU during training")
    plt.show()



# Save model + trainer state
save_path = "./gpe_translation_model"
trainer.save_model(save_path)


import shutil
from google.colab import files

# Replace 'my_folder' with your folder name
shutil.make_archive('gpe_translation_model', 'zip', 'gpe_translation_model')

# Download the zip
files.download('gpe_translation_model.zip')


# Pick a few sentences from validation
for i in range(5):
    en_sent = ds["validation"][i]["translation"]["en"]
    ref_si = ds["validation"][i]["translation"]["si"]

    # Encode English input
    input_ids = torch.tensor([tokenizer_en.encode(en_sent, max_length=64)]).to(model.device)

    # Generate translation
    output_ids = model.generate(input_ids, max_length=64)
    pred_si = tokenizer_si.decode(output_ids[0].tolist(), skip_special_tokens=True)

    print(f"EN: {en_sent}")
    print(f"REF: {ref_si}")
    print(f"PRED: {pred_si}")
    print("-" * 50)
