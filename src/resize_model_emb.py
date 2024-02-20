import os
import math
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(
                    prog='Resize model embedding')
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--tokenizer_path', type=str, required=True)
parser.add_argument('--export_dir', type=str, required=True)
parser.add_argument('--export_size', type=int, default=5)
args = parser.parse_args()

def _noisy_mean_initialization(embed_weight: torch.Tensor, num_new_tokens: int):
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(avg_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight


def _resize_embedding_layer(model, tokenizer) -> None:
    r"""
    Resize token embeddings.
    """
    current_embedding_size = model.get_input_embeddings().weight.size(0)
    if len(tokenizer) > current_embedding_size:
        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
            logger.warning("Current model does not support resizing token embeddings.")
            return

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        new_embedding_size = model.get_input_embeddings().weight.size(0)
        num_new_tokens = new_embedding_size - current_embedding_size
        _noisy_mean_initialization(model.get_input_embeddings().weight.data, num_new_tokens)
        _noisy_mean_initialization(model.get_output_embeddings().weight.data, num_new_tokens)

        print("Resized token embeddings from {} to {}.".format(current_embedding_size, new_embedding_size))


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    _resize_embedding_layer(model, tokenizer)

    model.config.use_cache = True
    model = model.to("cpu")
    print("Saving model...")
    model.save_pretrained(
        save_directory=args.export_dir,
        max_shard_size="{}GB".format(args.export_size),
        safe_serialization=True
    )

    try:
        print("Saving tokenizer...")
        tokenizer.padding_side = "left" # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(args.export_dir)
    except:
        print("Cannot save tokenizer, please copy the files manually.")


if __name__ == '__main__':
    main(args)