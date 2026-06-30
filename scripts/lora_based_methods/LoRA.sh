#!/bin/bash
# Ablation: LoRA baseline (no KD-tree, no OPL)
# total_loss = loss_ce only

now=$(date +"%m%d_%H%M%S")
gpu_nodes="0"
model_name="Llama-3.2-1B-Instruct"
epochs=2,1,3,2,1,2,2,3

# Train:
echo "Start training LoRA_Baseline..."
deepspeed --include=localhost:$gpu_nodes --master_port 25012 training/main.py \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path ./PTM/$model_name \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --num_train_epochs $epochs \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method LoRA_Baseline \
    --output_dir ./outputs_LLM-CL/cl/$model_name/LoRA_Baseline_$now \
    --reg 0.0 \
    --reg_orth 0.0

# Inference:
echo "Start inference..."
python inference/infer_multi_command.py \
    --gpus $gpu_nodes \
    --master_port 25012 \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path ./PTM/$model_name \
    --inference_model_path ./outputs_LLM-CL/cl/$model_name/LoRA_Baseline_$now \
    --inference_batch 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method LoRA_Baseline \
    --inference_output_path ./outputs_LLM-CL/cl/$model_name/LoRA_Baseline_$now/predictions

# Collect results:
echo "Start collecting results..."
python inference/collect_results.py \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --data_path ./outputs_LLM-CL/cl/$model_name/LoRA_Baseline_$now/predictions