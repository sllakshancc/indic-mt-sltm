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
tokenizer_si.save("tokenizer_si.json")