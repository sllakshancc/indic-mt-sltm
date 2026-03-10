import re
from collections import defaultdict
from tqdm import tqdm


class SuperBPETokenizer:
    def __init__(self, vocab_size=50000, transition_point=30000):
        """
        vocab_size: final vocabulary size (T)
        transition_point: vocab size at which we switch to Stage 2 (t)
        """
        assert transition_point < vocab_size
        self.vocab_size = vocab_size
        self.transition_point = transition_point

        self.vocab = {}          # id -> token
        self.vocab_re = {}       # token -> id
        self.merges = []         # ordered list of (pair, new_id)

        self.trained = False

        # whitespace regex (similar to GPT-2 style)
        self.whitespace_pattern = r"\S+|\s+"

    # =========================
    # Public Training Interface
    # =========================

    def train(self, texts):
        print("Stage 1: Learning subwords (whitespace enforced)")
        self._train_stage(texts, target_size=self.transition_point, enforce_whitespace=True)

        print("Stage 2: Learning superwords (whitespace disabled)")
        self._train_stage(texts, target_size=self.vocab_size, enforce_whitespace=False)

        self.trained = True
        print(f"Training complete. Final vocab size: {len(self.vocab)}")

    # =========================
    # Core Training Logic
    # =========================

    def _train_stage(self, texts, target_size, enforce_whitespace):
        if not self.vocab:
            self._initialize_vocab(texts, enforce_whitespace)

        ids_list = self._texts_to_ids(texts, enforce_whitespace)

        num_merges = target_size - len(self.vocab)

        for _ in tqdm(range(num_merges)):
            stats = self._get_stats(ids_list)

            if not stats:
                break

            pair = max(stats, key=stats.get)
            new_id = len(self.vocab)

            ids_list = [self._merge(ids, pair, new_id) for ids in ids_list]

            # record merge in order
            self.merges.append((pair, new_id))

            # update vocab
            token = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab[new_id] = token
            self.vocab_re[token] = new_id

    # =========================
    # Initialization
    # =========================

    def _initialize_vocab(self, texts, enforce_whitespace):
        tokens = set()

        for text in texts:
            chunks = self._pretokenize(text, enforce_whitespace)
            for chunk in chunks:
                for char in chunk:
                    tokens.add(char)

        for idx, token in enumerate(sorted(tokens)):
            self.vocab[idx] = token
            self.vocab_re[token] = idx

    # =========================
    # Pretokenization
    # =========================

    def _pretokenize(self, text, enforce_whitespace):
        if enforce_whitespace:
            return re.findall(self.whitespace_pattern, text)
        else:
            return [text]

    # =========================
    # Convert Text to IDs
    # =========================

    def _texts_to_ids(self, texts, enforce_whitespace):
        ids_list = []

        for text in texts:
            chunks = self._pretokenize(text, enforce_whitespace)

            for chunk in chunks:
                ids = []
                for char in chunk:
                    ids.append(self.vocab_re[char])
                if ids:
                    ids_list.append(ids)

        return ids_list

    # =========================
    # Pair Statistics
    # =========================

    def _get_stats(self, ids_list):
        stats = defaultdict(int)

        for ids in ids_list:
            for pair in zip(ids, ids[1:]):
                stats[pair] += 1

        return stats

    # =========================
    # Merge Operation
    # =========================

    def _merge(self, ids, pair, new_id):
        i = 0
        merged = []

        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1

        return merged

    # =========================
    # Encoding
    # =========================

    def encode(self, text):
        assert self.trained

        ids = []
        for char in text:
            if char in self.vocab_re:
                ids.append(self.vocab_re[char])
            else:
                continue

        # Apply merges in training order
        for pair, new_id in self.merges:
            ids = self._merge(ids, pair, new_id)

        return ids

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def tokenize(self, text):
        """Tokenize text into subword strings."""
        ids = self.encode(text)
        return [self.vocab.get(i, "<UNK>") for i in ids]