# Compatible package
# torch==1.13.1
# deepspeed==0.12.3
# flash-attn==2.4.1
# neptune
# autoawq: pip install https://github.com/casper-hansen/AutoAWQ/releases/download/v0.1.8/autoawq-0.1.8+cu118-cp310-cp310-linux_x86_64.whl
# ** If installing torch==2.1.0 ==> re-install DeepSpeed 0.12.6 & manually build flash-attn 2.4.1
# unsloth: pip install "unsloth[cu118_ampere] @ git+https://github.com/unslothai/unsloth.git"

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMTdmNWZlYy1lMTVjLTQyZWEtOTY5ZS1hOWM3ZmMyMjJjZTQifQ=="
export NEPTUNE_PROJECT="martinakaduc/VIURA"
export LIBRARY_PATH=/llm_quangduc/miniconda3/envs/mixsura2/lib/python3.10/site-packages/torch/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/llm_quangduc/miniconda3/envs/mixsura2/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
# Optional
# export OMP_NUM_THREADS=1

# Find links to system library of CUDA
# ldconfig -v 2>/dev/null | grep -v ^$'\t'

accelerate config
# - This machine
# - multi-GPU
# - node: 1
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
    --tokenizer_path tokenizers/ura-hcmut/MixSUra-tokenizer \
    --dataset wikipedia_vi \
    --batch_size 4096

python src/resize_model_emb.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tokenizer_path tokenizers/ura-hcmut/MixSUra-tokenizer_merged \
    --export_dir models/ura-hcmut/MixSUra-v0


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
    --dataset wikipedia_vi \
    --preprocessing_num_workers 32 \
    --cutoff_len 32768 \
    --num_train_epochs 1.0 \
    --max_samples 2000000 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 256 \
    --learning_rate 5e-05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.001 \
    --logging_steps 1 \
    --warmup_ratio 0.03 \
    --save_steps 2 \
    --neftune_noise_alpha 0 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --output_dir saves/MixSUra-wiki \
    --save_total_limit 5 \
    --plot_loss True


##### DPO #####
# In DPO batch_size should be divided by 2
# 8xA100 80GB #
accelerate launch src/train_bash.py \
    --stage dpo \
    --do_train True \
    --model_name_or_path models/ura-hcmut/MixSUra-wiki \
    --use_fast_tokenizer True \
    --finetuning_type lora \
    --template mistral \
    --flash_attn True \
    --dataset_dir data \
    --dataset orca_dpo_pairs_vi \
    --preprocessing_num_workers 32 \
    --cutoff_len 32768 \
    --num_train_epochs 1.0 \
    --max_samples 2000000 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 2 \ 
    --gradient_accumulation_steps 64 \
    --learning_rate 5e-05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.001 \
    --logging_steps 1 \
    --warmup_ratio 0.03 \
    --save_steps 2 \
    --neftune_noise_alpha 0 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --output_dir saves/MixSUra-orca-dpo \
    --save_total_limit 5 \
    --plot_loss True \
    --hf_hub_token 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 \
    --report_to none


# Test QKVO
# 4xA100 40GB #
#   --disable_gradient_checkpointing True \
# Set padding_side="left"
accelerate launch src/train_bash.py \
    --stage pt \
    --do_train True \
    --model_name_or_path models/ura-hcmut/MixSUra-qkvo-1 \
    --use_fast_tokenizer True \
    --finetuning_type lora \
    --template mistral \
    --flash_attn True \
    --dataset_dir data \
    --dataset news_corpus_vi \
    --preprocessing_num_workers 32 \
    --cutoff_len 32768 \
    --num_train_epochs 1.0 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 256 \
    --learning_rate 5e-05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.001 \
    --logging_steps 1 \
    --warmup_ratio 0.03 \
    --save_steps 2 \
    --neftune_noise_alpha 0 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1 \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --output_dir saves/MixSUra-qkvo-2 \
    --save_total_limit 5 \
    --plot_loss True \
    --report_to neptune

accelerate launch src/train_bash.py \
    --stage dpo \
    --do_train True \
    --model_name_or_path models/ura-hcmut/MixSUra-orca-dpo \
    --use_fast_tokenizer True \
    --finetuning_type lora \
    --template mistral \
    --flash_attn True \
    --dataset_dir data \
    --dataset orca_dpo_pairs_vi \
    --preprocessing_num_workers 32 \
    --cutoff_len 32768 \
    --num_train_epochs 1.0 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 256 \
    --learning_rate 5e-05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.001 \
    --logging_steps 1 \
    --warmup_ratio 0.03 \
    --save_steps 2 \
    --neftune_noise_alpha 0 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1 \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --output_dir saves/MixSUra-orca-dpo-qkvo \
    --save_total_limit 5 \
    --plot_loss True \
    --hf_hub_token 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 \
    --report_to neptune


## TEST FREEZE ##
accelerate launch src/train_bash.py \
    --stage pt \
    --do_train True \
    --model_name_or_path models/ura-hcmut/MixSUra-v0 \
    --use_fast_tokenizer True \
    --finetuning_type freeze-a2e \
    --template mistral \
    --flash_attn True \
    --dataset_dir data \
    --dataset wikipedia_vi \
    --preprocessing_num_workers 32 \
    --cutoff_len 32768 \
    --num_train_epochs 1.0 \
    --max_samples 2000000 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 128 \
    --learning_rate 5e-05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.001 \
    --logging_steps 1 \
    --warmup_ratio 0.03 \
    --save_steps 2 \
    --neftune_noise_alpha 0 \
    --name_module_trainable embed_tokens,lm_head \
    --output_dir MixSUra-v0.1 \
    --save_total_limit 5 \
    --plot_loss True
    

## EXPORT MODELS ##
python src/export_model.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --adapter_name_or_path saves/MixSUra-wiki/final \
    --use_fast_tokenizer True \
    --template mistral \
    --finetuning_type lora \
    --flash_attn True \
    --export_size 5 \
    --export_legacy_format False \
    --export_dir models/ura-hcmut/MixSUra-wiki

python src/export_model.py \
    --model_name_or_path models/ura-hcmut/MixSUra-wiki \
    --adapter_name_or_path saves/MixSUra-orca-dpo/final \
    --use_fast_tokenizer True \
    --template mistral \
    --finetuning_type lora \
    --flash_attn True \
    --export_size 5 \
    --export_legacy_format False \
    --export_dir models/ura-hcmut/MixSUra-orca-dpo

python src/export_model.py \
    --model_name_or_path models/ura-hcmut/MixSUra-orca-dpo \
    --adapter_name_or_path saves/MixSUra-qkvo/checkpoint-20 \
    --use_fast_tokenizer True \
    --template mistral \
    --finetuning_type lora \
    --flash_attn True \
    --export_size 5 \
    --export_legacy_format False \
    --export_dir models/ura-hcmut/MixSUra-qkvo

python src/export_model.py \
    --model_name_or_path models/ura-hcmut/MixSUra-qkvo \
    --adapter_name_or_path saves/MixSUra-qkvo-1/final \
    --use_fast_tokenizer True \
    --template mistral \
    --finetuning_type lora \
    --flash_attn True \
    --export_size 5 \
    --export_legacy_format False \
    --export_dir models/ura-hcmut/MixSUra-qkvo-1
    


## QUANTIZE MODELS ##
python src/quantize_model.py \
    --model_name_or_path models/ura-hcmut/MixSUra-qkvo-1 \
    --quantization_path models/ura-hcmut/MixSUra-AWQ


## DEPLOY MODELS ##
python src/web_demo.py \
    --model_name_or_path models/ura-hcmut/MixSUra-orca-dpo \
    --use_fast_tokenizer True \
    --template mistral \
    --flash_attn True

python src/cli_demo.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --adapter_name_or_path saves/Mixtral-8x7B-Chat/checkpoint-20 \
    --use_fast_tokenizer True \
    --template mistral \
    --flash_attn True \
    --finetuning_type lora

text-generation-launcher \
    --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --port 10025 \
    --max-input-length 28672 \
    --max-total-tokens 32768 \
    --max-batch-prefill-tokens 32768

text-generation-launcher \
    --model-id ./ \
    --port 10025 \
    --max-input-length 28672 \
    --max-total-tokens 32768 \
    --max-batch-prefill-tokens 32768


#################################
############ LLaMa-2 ############
#################################

# 4xA100 80GB
accelerate launch src/train_bash.py \
    --stage pt \
    --do_train True \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --use_fast_tokenizer True \
    --finetuning_type lora \
    --template llama2 \
    --flash_attn True \
    --dataset_dir data \
    --dataset wikipedia_vi \
    --preprocessing_num_workers 32 \
    --cutoff_len 4096 \
    --num_train_epochs 1.0 \
    --max_samples 2000000 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 256 \
    --learning_rate 5e-05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.001 \
    --logging_steps 1 \
    --warmup_ratio 0.03 \
    --save_steps 2 \
    --neftune_noise_alpha 0 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --output_dir saves/ura-llama-7b/wiki-test \
    --save_total_limit 10 \
    --plot_loss True
    
python src/export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --adapter_name_or_path saves/ura-llama-7b/wiki/final \
    --use_fast_tokenizer True \
    --flash_attn True \
    --template llama2 \
    --finetuning_type lora \
    --export_size 5 \
    --export_legacy_format False \
    --export_dir models/ura-hcmut/ura-llama-7b-wiki
    
# 4xA100 80GB
# --use_unsloth True \
accelerate launch src/train_bash.py \
    --stage dpo \
    --do_train True \
    --model_name_or_path models/ura-hcmut/ura-llama-7b-wiki \
    --use_fast_tokenizer True \
    --finetuning_type lora \
    --template llama2 \
    --flash_attn True \
    --dataset_dir data \
    --dataset orca_dpo_pairs_vi \
    --preprocessing_num_workers 32 \
    --cutoff_len 4096 \
    --num_train_epochs 5.0 \
    --max_samples 2000000 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1024 \
    --learning_rate 5e-05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.001 \
    --logging_steps 1 \
    --warmup_ratio 0.03 \
    --save_steps 2 \
    --neftune_noise_alpha 0 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1 \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --output_dir saves/ura-llama-7b/dpo-qkvo \
    --save_total_limit 10 \
    --hf_hub_token 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 \
    --plot_loss True
    
python src/export_model.py \
    --model_name_or_path models/ura-hcmut/ura-llama-7b-wiki \
    --adapter_name_or_path saves/ura-llama-7b/dpo/final \
    --use_fast_tokenizer True \
    --flash_attn True \
    --template llama2 \
    --finetuning_type lora \
    --export_size 5 \
    --export_legacy_format False \
    --export_dir models/ura-hcmut/ura-llama-7b-dpo
    
python src/cli_demo.py \
    --model_name_or_path models/ura-hcmut/ura-llama-7b-dpo \
    --template llama2 \
    --use_fast_tokenizer True \
    --flash_attn True

python src/web_demo.py \
    --model_name_or_path models/ura-hcmut/ura-llama-7b-dpo \
    --template llama2 \
    --use_fast_tokenizer True \
    --flash_attn True