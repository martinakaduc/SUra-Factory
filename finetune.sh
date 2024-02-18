# Compatible package
# Python 3.10
# torch==2.1.0+cu118
# deepspeed==0.13.1
# flash-attn==2.4.1
# neptune
# autoawq: pip install https://github.com/casper-hansen/AutoAWQ/releases/download/v0.1.8/autoawq-0.1.8+cu118-cp310-cp310-linux_x86_64.whl
# unsloth: pip install "unsloth[cu118_ampere] @ git+https://github.com/unslothai/unsloth.git"

export NEPTUNE_API_TOKEN=""
export NEPTUNE_PROJECT=""
export LIBRARY_PATH=<path_to_env>/lib/python3.10/site-packages/torch/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=<path_to_env>/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# Optional
# export OMP_NUM_THREADS=1

# Find links to system library of CUDA
# ldconfig -v 2>/dev/null | grep -v ^$'\t'

# CONFIG MACHINE(s)
accelerate config
# - This machine
# - multi-GPU
# - node: 1  # Number of nodes
# - check errors: NO
# - torch dynamo: NO
# - DeepSpeed: yes
# - DeepSpeed file: NO
# - ZeRO: 3
# - Offload optimzier: cpu
# - Offload parameters: cpu
# - Gradient accumulation: [based_on_command]
# - Gradient clipping: [based_on_command]
# - Clipping value: [based_on_command]
# - Save 16-bit: yes
# - deepspeed.zero.Init: yes
# - Number of GPUs: [Number_available_GPUs]
# - Dtype: BF16


## TRAINING NEW TOKENIZER ##
python src/train_tokenizer.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tokenizer_path tokenizers/ura-hcmut/Mixtral-tokenizer \
    --dataset wikipedia_vi \
    --batch_size 4096

python src/resize_model_emb.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tokenizer_path tokenizers/ura-hcmut/Mixtral-tokenizer_merged \
    --export_dir models/ura-hcmut/Mixtral-ViTokens


## PRE-TRAINING ##
## 4xA100 40GB ###
accelerate launch src/train_bash.py \
    --stage pt \
    --do_train True \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --use_fast_tokenizer True \
    --finetuning_type lora \
    --template mistral \
    --flash_attn True \
    --dataset_dir data \
    --dataset wikipedia_vi,literature_vi \
    --preprocessing_num_workers 32 \
    --cutoff_len 32768 \
    --num_train_epochs 5.0 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 256 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.001 \
    --logging_steps 1 \
    --warmup_ratio 0.02 \
    --save_steps 2 \
    --neftune_noise_alpha 0 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1 \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --output_dir saves/MixSUra \
    --save_total_limit 3 \
    --plot_loss True \
    --report_to neptune


##### SFT #####
# 4xA100 40GB #
accelerate launch src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path models/ura-hcmut/MixSUra \
    --use_fast_tokenizer True \
    --finetuning_type lora \
    --template mistral \
    --flash_attn True \
    --dataset_dir data \
    --dataset OPUS100-vien-dpo,OPUS100-envi-dpo,PhoMT-vien-dpo,PhoMT-envi-dpo,vietnews-dpo,wiki_lingua-dpo,zalo_e2eqa-dpo,VSEC-dpo,10vancauhoi-sft \
    --preprocessing_num_workers 32 \
    --mix_strategy interleave_over \
    --num_train_epochs 2.0 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.001 \
    --logging_steps 1 \
    --warmup_ratio 0.02 \
    --save_steps 2 \
    --neftune_noise_alpha 0 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1 \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --output_dir saves/MixSUra-SFT \
    --save_total_limit 3 \
    --plot_loss True \
    --hf_hub_token <hf_hub_token> \
    --report_to neptune
    

## EXPORT MODELS ##
python src/export_model.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --adapter_name_or_path saves/MixSUra \
    --use_fast_tokenizer True \
    --template mistral \
    --export_size 5 \
    --export_legacy_format False \
    --export_dir models/ura-hcmut/MixSUra \
    --export_hub_model_id ura-hcmut/MixSUra \
    --hf_hub_token <hf_hub_token>

python src/export_model.py \
    --model_name_or_path models/ura-hcmut/MixSUra \
    --adapter_name_or_path saves/MixSUra-SFT \
    --use_fast_tokenizer True \
    --template mistral \
    --export_size 5 \
    --export_legacy_format False \
    --export_dir models/ura-hcmut/MixSUra-SFT \
    --export_hub_model_id ura-hcmut/MixSUra-SFT \
    --hf_hub_token <hf_hub_token>


## QUANTIZE MODELS ##
# AWQ
python src/quantize_model.py \
    --model_name_or_path models/ura-hcmut/MixSUra \
    --quantization_path models/ura-hcmut/MixSUra-AWQ

python src/quantize_model.py \
    --model_name_or_path models/ura-hcmut/MixSUra-SFT \
    --quantization_path models/ura-hcmut/MixSUra-SFT-AWQ

# LLAMA.CPP
python src/llama.cpp/convert.py models/ura-hcmut/MixSUra --outtype f16 --outfile models/ura-hcmut/MixSUra/model.bin
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q4_0.bin q4_0
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q2_K.bin q2_K
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q3_K.bin q3_K
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q3_K_S.bin q3_K_S
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q3_K_M.bin q3_K_M
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q3_K_L.bin q3_K_L
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q4_1.bin q4_1
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q4_K.bin q4_K
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q4_K_S.bin q4_K_S
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q4_K_M.bin q4_K_M
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q5_0.bin q5_0
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q5_1.bin q5_1
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q5_K.bin q5_K
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q5_K_S.bin q5_K_S
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q5_K_M.bin q5_K_M
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q6_K.bin q6_K
src/llama.cpp/quantize models/ura-hcmut/MixSUra/model.bin models/ura-hcmut/MixSUra/model-q8_0.bin q8_0

python src/llama.cpp/convert.py models/ura-hcmut/MixSUra-SFT --outtype f16 --outfile models/ura-hcmut/MixSUra-SFT/model.bin
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q4_0.bin q4_0
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q2_K.bin q2_K
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q3_K.bin q3_K
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q3_K_S.bin q3_K_S
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q3_K_M.bin q3_K_M
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q3_K_L.bin q3_K_L
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q4_1.bin q4_1
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q4_K.bin q4_K
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q4_K_S.bin q4_K_S
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q4_K_M.bin q4_K_M
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q5_0.bin q5_0
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q5_1.bin q5_1
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q5_K.bin q5_K
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q5_K_S.bin q5_K_S
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q5_K_M.bin q5_K_M
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q6_K.bin q6_K
src/llama.cpp/quantize  models/ura-hcmut/MixSUra-SFT/model.bin models/ura-hcmut/MixSUra-SFT/model-q8_0.bin q8_0


llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q4_0.bin q4_0
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q2_K.bin q2_K
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q3_K.bin q3_K
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q3_K_S.bin q3_K_S
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q3_K_M.bin q3_K_M
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q3_K_L.bin q3_K_L
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q4_1.bin q4_1
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q4_K.bin q4_K
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q4_K_S.bin q4_K_S
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q4_K_M.bin q4_K_M
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q5_0.bin q5_0
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q5_1.bin q5_1
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q5_K.bin q5_K
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q5_K_S.bin q5_K_S
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q5_K_M.bin q5_K_M
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q6_K.bin q6_K
llm/llama.cpp/quantize  MixSUra-SFT.bin MixSUra-SFT-q8_0.bin q8_0


## DEPLOY MODELS ##
python src/web_demo.py \
    --model_name_or_path models/ura-hcmut/MixSUra \
    --use_fast_tokenizer True \
    --template mistral \
    --flash_attn True

python src/cli_demo.py \
    --model_name_or_path models/ura-hcmut/MixSUra \
    --use_fast_tokenizer True \
    --template mistral \
    --flash_attn True

# OR USING TGI
text-generation-launcher \
    --model-id models/ura-hcmut/MixSUra \
    --port 10025 \
    --max-input-length 28672 \
    --max-total-tokens 32768 \
    --max-batch-prefill-tokens 28672

# OR USING OLLAMA
ollama run MixSUra
