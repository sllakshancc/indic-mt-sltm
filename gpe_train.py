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


        # Regex pattern for text chunking
        #self.whitespace_pattern = r"\w+|[^\w\s]"
        self.whitespace_pattern = r" ?\w+| ?[^\w\s]+"


    def train(self, texts):
        """Train tokenizer on list of text strings."""
        print(f"Training GPE tokenizer on {len(texts)} texts...")

        # Initialize vocab with special tokens
        self.vocab = {v: k for k, v in self.special_tokens.items()}
        self.vocab_re = self.special_tokens.copy()


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
















































