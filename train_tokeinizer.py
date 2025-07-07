from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

ds = load_dataset("Helsinki-NLP/opus-100", "en-si", cache_dir="./hf_cache")

train_texts_si = [ex["si"] for ex in ds["train"]["translation"]]



# Sinhala tokenizer
tokenizer_si = Tokenizer(models.BPE())
tokenizer_si.pre_tokenizer = pre_tokenizers.Whitespace()
trainer_si = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"], vocab_size=50)
tokenizer_si.train_from_iterator(train_texts_si[:10], trainer=trainer_si)

# Save tokenizers
#tokenizer_si.save("tokenizer_si.json")

import sys
sys.path.insert(0, 'tokenizers')
from GPETokenizer import GPETokenizer

# 1. Create and train a tokenizer
tokenizer = GPETokenizer(vocab_size=4000, dummy_prefix=None)

# Train the tokenizer
tokenizer.train(train_texts_si[:100000])

# 2. Encode text to token IDs
text = "මම ගෙදර යනවා."
token_ids = tokenizer.encode(text)
print(f"Encoded: {token_ids}")

# 3. Decode token IDs back to text
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")

# 4. Tokenize text into subword strings
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# 6. Save and load the tokenizer
tokenizer.save_pretrained_tokenizer_json("my_tokenizer.json")  # Saves vocab and config

from transformers import PreTrainedTokenizerFast

tok_en = PreTrainedTokenizerFast(tokenizer_file="my_tokenizer.json", bos_token="[BOS]", eos_token="[EOS]",
                                 unk_token="[UNK]", pad_token="[PAD]")


text = "මම ගෙදර යනවා."
token_ids = tok_en.encode(text)
print(f"Encoded: {token_ids}")

# 3. Decode token IDs back to text
decoded_text = tok_en.decode(token_ids)
print(f"Decoded: {decoded_text}")

# 4. Tokenize text into subword strings
tokens = tok_en.tokenize(text)
print(f"Tokens: {tokens}")