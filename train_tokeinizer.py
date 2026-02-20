import argparse
import os
from datasets import load_dataset
from tokenizers.GPETokenizer import GPETokenizer


def parse_args():
	parser = argparse.ArgumentParser(description="Train GPE tokenizer from a dataset")
	parser.add_argument("--subset", type=int, default=None, help="Number of training examples to use (omit to use full dataset)")
	parser.add_argument("--name", type=str, default="gpe_test", help="output tokenizer name")
	parser.add_argument("--dataset", type=str, default="Helsinki-NLP/opus-100", help="HuggingFace dataset id")
	parser.add_argument("--dataset-pair", dest="dataset_pair", type=str, default="en-si", help="dataset language pair")
	parser.add_argument("--cache-dir", type=str, default="./hf_cache", help="HF cache dir")
	parser.add_argument("--output-dir", type=str, default="./tokenizers_trained", help="where to save the tokenizer")
	parser.add_argument("--vocab-size", type=int, default=4000, help="vocabulary size for the tokenizer")
	return parser.parse_args()


def main():
	args = parse_args()

	ds = load_dataset(args.dataset, args.dataset_pair, cache_dir=args.cache_dir)
	train_texts_si = [ex["si"] for ex in ds["train"]["translation"]]

	if args.subset is not None:
		if args.subset > 0:
			train_texts = train_texts_si[: args.subset]
		else:
			train_texts = train_texts_si
	else:
		train_texts = train_texts_si

	tokenizer = GPETokenizer(vocab_size=args.vocab_size)
	tokenizer.train(train_texts)

	out_path = os.path.join(args.output_dir, args.name)
	tokenizer.save(out_path)
	print(f"Tokenizer saved to: {out_path}")


if __name__ == "__main__":
	main()