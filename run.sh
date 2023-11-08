

# python sft.py \
#     --model_name meta-llama/Llama-2-7b-chat-hf \
#     --dataset_name timdettmers/openassistant-guanaco \
#     --load_in_4bit \
#     --use_peft \
#     --num_train_epochs 1 \
#     --peft_lora_r 8 \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --save_strategy steps \
#     --eval_steps 100 \
#     --load_best_model_at_end True \
#     --save_total_limit 2 \

CUDA_VISIBLE_DEVICES=0 python sft.py \
    --do_train \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name timdettmers/openassistant-guanaco \
    --load_in_8bit \
    --use_peft \
    --gradient_checkpointing True \
    --peft_lora_r 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --output_dir output