export CUDA_VISIBLE_DEVICES=2
python ../model/bert/bert_main.py  --model_save_dir='/opt/hyp/NER/NER-model/saved_models/msra_bert/' \
                     --tensorboard_dir='/opt/hyp/NER/NER-model/saved_models/msra_bert/runs/' \
                     --batch_size=16 \
                     --data_path='/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data' \
                     --model_name_or_path='/opt/hyp/NER/embedding/bert/chinese_L-12_H-768_A-12_pytorch' \
                     --num_train_epochs=10 \
                     --do_train=True \
                     --seed=42 \
                     --max_seq_length=156 \
                     --use_bieos=True \
                     --learning_rate=5e-5 \
                     --use_dataParallel=False \
                     --use_crf=False \





