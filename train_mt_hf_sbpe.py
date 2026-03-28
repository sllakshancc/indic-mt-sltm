import os
import torch
from datasets import load_dataset
import numpy as np
from transformers import (
    EncoderDecoderModel,
    EncoderDecoderConfig,
    BertConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import evaluate

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split, ByteLevel, Whitespace, Metaspace
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence

bleu = evaluate.load("sacrebleu")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")


PAIR = "en-si"
SRC_LANG = "si"
TGT_LANG = "en"

MAX_LEN = 128
PAD_ID = 32001
UNK_ID = 32002
BOS_ID = 32003
EOS_ID = 32004

VOCAB_SIZE_SI = 32005
VOCAB_SIZE_TA = 32005

print("Loading tokenizers...")
tokenizer_si = Tokenizer.from_file("./tokenizers_trained/superbpe_opus_100_si_10G_20K_extend_32K/tokenizer.json")
tokenizer_en = Tokenizer.from_file("./tokenizers_trained/superbpe_opus_100_en_10G_20K_extend_32K/tokenizer.json")

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
    config.bos_token_id = BOS_ID
    config.pad_token_id = PAD_ID
    config.eos_token_id = EOS_ID
    #config.forced_eos_token_id = EOS_ID
    config.max_length = MAX_LEN
    config.num_beams = 4
    config.early_stopping = True

    #check these settings, as they can cause issues if not set correctly
    config.tie_encoder_decoder = False
    config.tie_word_embeddings = False

    model = EncoderDecoderModel(config)

    return model


# =========================
# DATASET PREPROCESS
# =========================

def preprocess(example):

    src = example["translation"][SRC_LANG]
    tgt = example["translation"][TGT_LANG]

    src_ids = tokenizer_si.encode(src)[:MAX_LEN-2]
    tgt_ids = tokenizer_en.encode(tgt)[:MAX_LEN-2]

    src_ids = [BOS_ID] + src_ids + [EOS_ID]
    tgt_ids = [BOS_ID] + tgt_ids + [EOS_ID]

    return {
        "input_ids": src_ids,
        "labels": tgt_ids
    }


def collate_fn(batch):

    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=PAD_ID
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100    #check if -100 is the correct padding value for labels in seq2seq models
    )

    # Create decoder_input_ids (The "Shift Right" fix)
    decoder_input_ids = labels.new_zeros(labels.shape)
    decoder_input_ids[:, 1:] = labels[:, :-1].clone()
    decoder_input_ids[:, 0] = BOS_ID
    decoder_input_ids.masked_fill_(decoder_input_ids == -100, PAD_ID)
    # While optional in some versions, it's best practice to provide this
    decoder_attention_mask = (decoder_input_ids != PAD_ID).long()

    attention_mask = (input_ids != PAD_ID).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels
    }


def compute_metrics(eval_preds):

    preds, labels = eval_preds

    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in labels so we can decode them
    labels = np.where(labels != -100, labels, PAD_ID)

    decoded_preds = [
        tokenizer_en.decode(p, skip_special_tokens=True)
        for p in preds
    ]

    decoded_labels = [
        tokenizer_en.decode(l, skip_special_tokens=True)
        for l in labels
    ]

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = bleu.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    return {"bleu": result["score"]}


# =========================
# MAIN
# =========================

print("Loading dataset...")
dataset = load_dataset("Helsinki-NLP/opus-100", PAIR, cache_dir="./hf_cache")

print("Preprocessing dataset...")
dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names, num_proc=8)

#DEBUG: Create a tiny slice of the data
small_train_dataset = dataset["train"].select(range(100))
small_eval_dataset = dataset["validation"].select(range(50))


print("Building model...")
model = build_model()

print("creating training arguments...")
training_args = Seq2SeqTrainingArguments(

    output_dir="./checkpoints",
    save_total_limit=2,
    metric_for_best_model="bleu",
    greater_is_better=True,
    load_best_model_at_end=True,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps = 2,

    num_train_epochs=8,

    learning_rate=3e-4,

    #warmup_steps=1000,
    warmup_ratio = 0.1,

    max_grad_norm=1.0,

    fp16=True,

    eval_strategy="epoch",
    save_strategy="epoch",

    logging_steps=100,

    #label_smoothing_factor=0.1,

    predict_with_generate=True,

    push_to_hub=False,

    report_to="none"
)

print("Creating trainer...")
trainer = Seq2SeqTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],
    #train_dataset=small_train_dataset,

    eval_dataset=dataset["validation"],
    #eval_dataset=small_eval_dataset,

    data_collator=collate_fn,

    compute_metrics=compute_metrics
)

print("Starting training...")
if os.path.exists("./super_bpe/checkpoints") and len(os.listdir("./super_bpe/checkpoints")) > 0:
    print("Resuming from checkpoint...")
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

print("Saving final model...")
trainer.save_model("./super_bpe/final_translation_model")