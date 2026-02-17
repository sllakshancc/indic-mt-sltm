import os
import re
import time
import pickle
import regex
import grapheme
from tqdm.auto import tqdm
import json


class GPETokenizer:
    """GPE Tokenizer with HuggingFace-compatible interface"""

    def __init__(self, vocab_size=5000, dummy_prefix=" "):
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
        for text in tqdm(texts, desc="Collecting graphemes"):  # Limit for speed texts[:10000]
            text_chunks = regex.findall(self.whitespace_pattern, text)
            text_chunks = [t.replace(' ','\u2581') for t in text_chunks if t.strip()]

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
        num_merges = self.vocab_size - len(self.vocab)

        # Convert texts to IDs for merging
        ids_list = self._convert_to_ids_train(texts)

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

    def save_pretrained_tokenizer_json(self, path="tokenizer.json"):
        """
        Save this GPE tokenizer in HuggingFace PreTrainedTokenizerFast-compatible JSON format
        """

        if not self.trained:
            raise ValueError("Tokenizer must be trained before saving.")

        # -------------------------
        # Build vocab: token -> id
        # -------------------------
        #vocab_token_to_id = {token: idx for idx, token in self.vocab.items()}

        # -------------------------
        # Build merges: list of [token1, token2]
        # Order matters → sort by merged id
        # -------------------------
        # merges = []
        # for (id1, id2), new_id in sorted(self.merges.items(), key=lambda x: x[1]):
        #     merges.append([
        #         self.vocab[id1],
        #         self.vocab[id2]
        #     ])

        # merges are inserted in training order
        merges = [
            [self.vocab[id1], self.vocab[id2]]
            for (id1, id2), _ in self.merges.items()
        ]

        # -------------------------
        # Added special tokens
        # -------------------------
        added_tokens = []
        for token, idx in self.special_tokens.items():
            added_tokens.append({
                "id": idx,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            })

        # -------------------------
        # Final tokenizer.json
        # -------------------------
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": None,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": self.unk_token,
                "continuing_subword_prefix": self.dummy_prefix,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": False,
                "ignore_merges": False,
                "vocab": self.vocab_re,
                "merges": merges
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

        print(f"Tokenizer saved to {path}")

    def encode(self, text, add_special_tokens=True, max_length=None, truncation=False, padding=False):
        """Encode text to token IDs."""
        if not self.trained:
            raise ValueError("Tokenizer not trained")

        # Handle batch input
        if isinstance(text, list):
            return [self.encode(t, add_special_tokens, max_length, truncation, padding) for t in text]

        # Process text
        text_chunks = regex.findall(self.whitespace_pattern, text)
        text_chunks = [t.replace(' ','\u2581') for t in text_chunks if t.strip()]

        ids = []
        for chunk in text_chunks:
            graphemes_list = list(grapheme.graphemes(chunk))
            for g in graphemes_list:
                if g in self.vocab_re:
                    ids.append(self.vocab_re[g])
                else:
                    ids.append(self.unk_token_id)

        # Apply merges
        ids = self._apply_merges(ids)

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
        for i in ids:
            if skip_special_tokens and i in [0, 1, 2, 3]:
                continue
            if i in self.vocab:
                tokens.append(self.vocab[i])

        text = "".join(tokens).replace("▁", " ")
        return text.strip()

    def batch_decode(self, ids_batch, skip_special_tokens=True):
        """Decode batch of token IDs."""
        return [self.decode(ids, skip_special_tokens) for ids in ids_batch]

    def __len__(self):
        """Return vocabulary size."""
        return len(self.vocab)

    def tokenize(self, text):
        """Tokenize text into subword strings."""
        ids = self.encode(text, add_special_tokens=False)
        return [self.vocab.get(i, self.unk_token) for i in ids]


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
        # made_merge = True
        # while made_merge:
        #     made_merge = False
        #     new_ids = []
        #     i = 0
        #     while i < len(ids):
        #         if i < len(ids) - 1 and (ids[i], ids[i+1]) in self.merges:
        #             new_ids.append(self.merges[(ids[i], ids[i+1])])
        #             i += 2
        #             made_merge = True
        #         else:
        #             new_ids.append(ids[i])
        #             i += 1
        #     ids = new_ids
        # return ids

        for pair in self.merges.keys():
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def _convert_to_ids_train(self, texts):
        """Convert texts to IDs for training."""
        ids_list = []
        for text in texts:
            text_chunks = regex.findall(self.whitespace_pattern, text)
            text_chunks = [t.replace(' ','\u2581') for t in text_chunks if t.strip()]

            for chunk in text_chunks:
                graphemes_list = list(grapheme.graphemes(chunk))
                chunk_ids = []
                for g in graphemes_list:
                    if g in self.vocab_re:
                        chunk_ids.append(self.vocab_re[g])
                if chunk_ids:
                    ids_list.append(chunk_ids)

        return ids_list