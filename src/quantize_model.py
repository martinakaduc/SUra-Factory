import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

parser = argparse.ArgumentParser(
                    prog='Model Quantization',
                    description='Quantize a LLM')
parser.add_argument('--model_name_or_path', type=str, required=True) 
parser.add_argument('--quantization_path', type=str, required=True)
parser.add_argument('--quantization_bit', type=int, default=4)
parser.add_argument('--quantization_group_size', type=int, default=128)
args = parser.parse_args()

model_path = args.model_name_or_path
quant_path = args.quantization_path
modules_to_not_convert = ["gate"]
quant_config = {
    "zero_point": True, "q_group_size": args.quantization_group_size,
    "w_bit": args.quantization_bit, "version": "GEMM",
    "modules_to_not_convert": modules_to_not_convert
}

# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoAWQForCausalLM.from_pretrained(
    model_path, safetensors=True, **{"low_cpu_mem_usage": True}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def load_wikitext():
    data = load_dataset('wikimedia/wikipedia', '20231101.vi', split="train")
    return [text for text in data["text"] if text.strip() != '']

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    modules_to_not_convert=modules_to_not_convert,
    calib_data=load_wikitext()
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')