#!/bin/bash
# Script để chạy Tree_LoRA_Ortho (Tree_LoRA + Orthogonal Projection Loss)

now=$(date +"%m%d_%H%M%S")
gpu_nodes="0"
model_name="Llama-3.2-1B-Instruct"
epochs=2,1,3,2,1,2,2,3

reg=0.5
reg_orth=0.1      # lambda cho Orthogonal Projection Loss
vis_interval=50   # ghi log gradient mỗi bao nhiêu step

# Train:
echo "Start training..."
deepspeed --include=localhost:$gpu_nodes --master_port 25011 training/main.py \
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
    --CL_method Tree_LoRA_Ortho \
    --output_dir ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_Ortho_$now \
    --reg $reg \
    --reg_orth $reg_orth \
    --vis_interval $vis_interval

# Inference:
echo "Start inference..."
python inference/infer_multi_command.py \
    --gpus $gpu_nodes \
    --master_port 25011 \
    --data_path ./data/LLM-CL-Benchmark/LLM-CL-Benchmark_500 \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path ./PTM/$model_name \
    --inference_model_path ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_Ortho_$now \
    --inference_batch 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method Tree_LoRA_Ortho \
    --inference_output_path ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_Ortho_$now/predictions

# Collect results:
echo "Start collecting results..."
python inference/collect_results.py \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --data_path ./outputs_LLM-CL/cl/$model_name/Tree_LoRA_Ortho_$now/predictions