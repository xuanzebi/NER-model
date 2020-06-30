export CUDA_VISIBLE_DEVICES=0
python  ./examples/run_glue.py --data_dir='/opt/hyp/class_bert/data_json/trainans_data_1_1.json' \
            --model_type=bert \
            --model_name_or_path=/opt/hyp/class_bert/embedding/bert/chinese_L-12_H-768_A-12_pytorch \
            --task_name=clss \
            --output_dir=/opt/hyp/class_bert/outputs/qiye_chat1_1_3epoch \
            --max_seq_length=32 \
            --do_train=True \
            --do_eval=True \
            --do_test=False \
            --per_gpu_train_batch_size=8 \
            --per_gpu_eval_batch_size=16 \
            --learning_rate=5e-5 \
            --num_train_epochs=3 \

