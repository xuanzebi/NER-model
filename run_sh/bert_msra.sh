export CUDA_VISIBLE_DEVICES=2
python ../model/bert/bert_main.py  --model_save_dir='/opt/hyp/NER/NER-model/saved_models/msra/msra_bert_mrc/' \
                     --tensorboard_dir='/opt/hyp/NER/NER-model/saved_models/msra/msra_bert_mrc/runs/' \
                     --batch_size=16 \
                     --data_path='/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data' \
                     --model_name_or_path='/opt/hyp/NER/embedding/bert/chinese_L-12_H-768_A-12_pytorch' \
                     --num_train_epochs=10 \
                     --do_train=True \
                     --do_test=True \
                     --seed=42 \
                     --max_seq_length=156 \
                     --use_bieos=True \
                     --learning_rate=5e-6 \
                     --use_dataParallel=False \
                     --model_class='bert_mrc' \
                     --use_crf=False \
                     --data_type='zh_msra_ner' \








