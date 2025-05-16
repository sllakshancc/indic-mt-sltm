import os
import re
import time
import pickle
import regex
import grapheme
import tqdm
import json


class GPETokenizer:
    """GPE Tokenizer with HuggingFace-compatible interface"""

    def __init__(self, vocab_size=5000, dummy_prefix="▁"):
        self.vocab_size = vocab_size
        self.dummy_prefix = dummy_prefix

        self.vocab = {}       # id -> string
        self.vocab_re = {}    # string -> id
        self.merges = {}      # (id1, id2) -> new_id
        self.trained = False

        # Regex pattern for text chunking
        #self.whitespace_pattern = r"\w+|[^\w\s]"
        self.whitespace_pattern = r" ?\w+| ?[^\w\s]+"
