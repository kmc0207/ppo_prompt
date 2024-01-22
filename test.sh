dataset_name='sst2'
project_name='APE_test'
run_name='test'
CUDA_VISIBLE_DEVICES=3,4,5,6 python spo_v2.py \
    --agent_model_name "teknium/OpenHermes-2.5-Mistral-7B" \
    --dataset_name $dataset_name \
    --run_name 'test' \
    --max_length 125 \
    --learning_rate 0.00000001 \
    --test_term 5 \
    --train_batch_size 16 \
    --project $project_name \
    --max_epochs 50 \
    --queue_size 5 \
    --fewshot True \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \