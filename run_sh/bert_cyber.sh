export CUDA_VISIBLE_DEVICES=0,1,2
python ../model/bert/bert_main.py  --model_save_dir='/opt/hyp/NER/NER-model/saved_models/cyber/cyber_bert_parallel/' \
                     --tensorboard_dir='/opt/hyp/NER/NER-model/saved_models/cyber/cyber_bert_parallel/runs/' \
                     --batch_size=64 \
                     --data_path='/opt/hyp/NER/NER-model/data/json_data' \
                     --model_name_or_path='/opt/hyp/NER/embedding/bert/chinese_L-12_H-768_A-12_pytorch' \
                     --num_train_epochs=50 \
                     --do_train=True \
                     --do_test=True \
                     --seed=42 \
                     --max_seq_length=200 \
                     --use_bieos=True \
                     --learning_rate=5e-5 \
                     --use_dataParallel=True \
                     --use_crf=False \
                     --model_class='bert' \






