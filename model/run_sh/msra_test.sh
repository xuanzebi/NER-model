export CUDA_VISIBLE_DEVICES=1
python ../main.py --use_scheduler=True \
                     --model_save_dir='/opt/hyp/NER/NER-model/saved_models/test_msra/' \
                     --tensorboard_dir='/opt/hyp/NER/NER-model/saved_models/test_msra/runs/' \
                     --batch_size=128 \
                     --optimizer='Adam' \
                     --momentum=0.9 \
                     --data_path='/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data' \
                     --dump_embedding=True \
                     --save_embed_path='/opt/hyp/NER/NER-model/data/embedding/sgns.baidubaike.bigram-char_msra.p' \
                     --pred_embed_path='/opt/hyp/NER/embedding/sgns.baidubaike.bigram-char' \
                     --load=True \
                     --num_train_epochs=30 \
                     --do_train=True \
                     --word_emb_dim=300 \
                     --max_seq_length=156 \
                     --deal_long_short_data='pad' \






