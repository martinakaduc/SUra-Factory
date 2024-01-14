import argparse
import json
import os
import shutil
from datasets import load_dataset
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(
                    prog='Training new tokenizer')
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--tokenizer_path', type=str, required=True)
parser.add_argument('--dataset', type=str, default='wikimedia/wikipedia')
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--max_vocab_size', type=int, default=64000)
args = parser.parse_args()


def get_training_corpus(dataset_name, batch_size=1024):
    if dataset_name == 'wikipedia_vi':
        dataset = load_dataset('wikimedia/wikipedia', '20231101.vi', split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")
        
    for i in range(0, len(dataset), batch_size):
        yield [x.split("Liên kết ngoài")[0].split("Xem thêm")[0].split("Tham khảo")[0].split("Chú thích")[0].strip()
            for x in dataset[i : i + batch_size]["text"]]

def main(args):
    old_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    current_vocab_size = len(old_tokenizer.vocab)
    maximum_new_tokens = args.max_vocab_size - current_vocab_size
    
    text_iterator = get_training_corpus(args.dataset, batch_size=args.batch_size)
    new_tokenizer = old_tokenizer.train_new_from_iterator(text_iterator, maximum_new_tokens)
    
    old_tokenizer.save_pretrained(args.tokenizer_path + "_original")
    new_tokenizer.save_pretrained(args.tokenizer_path)
    
    # Merging tokenizer
    with open(args.tokenizer_path + "_original/tokenizer.json", "r", encoding="utf8") as f:
        old_tokenizer_json = json.load(f)

    with open(args.tokenizer_path + "/tokenizer.json", "r", encoding="utf8") as f:
        new_tokenizer_json = json.load(f)

    list_old_vocab = list(old_tokenizer_json["model"]["vocab"].keys())
    old_vocab_size = len(list_old_vocab)
    list_new_vocab = list(new_tokenizer_json["model"]["vocab"].keys())

    new_tokens = list(set(list_new_vocab).difference(list_old_vocab))
    for i, token in enumerate(new_tokens):
        old_tokenizer_json["model"]["vocab"][token] = old_vocab_size + i

    new_merges = list(set(old_tokenizer_json["model"]["merges"]).difference(new_tokenizer_json["model"]["merges"]))
    old_tokenizer_json["model"]["merges"] = new_tokenizer_json["model"]["merges"] + new_merges
    
    os.mkdir(args.tokenizer_path + "_merged")
    with open(args.tokenizer_path + "_merged/tokenizer.json", "w", encoding="utf8") as f:
        json.dump(old_tokenizer_json, f, indent=4, ensure_ascii=False)

    shutil.copy(args.tokenizer_path + "_original/tokenizer_config.json", args.tokenizer_path + "_merged/tokenizer_config.json")
    shutil.copy(args.tokenizer_path + "_original/special_tokens_map.json", args.tokenizer_path + "_merged/special_tokens_map.json")

if __name__ == '__main__':
    main(args)