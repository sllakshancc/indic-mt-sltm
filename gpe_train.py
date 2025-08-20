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


















































