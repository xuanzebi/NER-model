export CUDA_VISIBLE_DEVICES=0
python ../main.py --use_scheduler=True \
                     --model_save_dir='/opt/hyp/NER/NER-model/saved_models/test_weibo/' \
                     --tensorboard_dir='/opt/hyp/NER/NER-model/saved_models/test_weibo/runs/' \
                     --batch_size=64 \
                     --optimizer='Adam' \
                     --momentum=0.9 \
                     --data_path='/opt/hyp/NER/NER-model/data/other_data/WeiboNER/json_data' \
                     --dump_embedding=True \
                     --save_embed_path='/opt/hyp/NER/NER-model/data/embedding/sgns.baidubaike.bigram-char_weibo.p' \
                     --pred_embed_path='/opt/hyp/NER/embedding/sgns.baidubaike.bigram-char' \
                     --load=True \
                     --num_train_epochs=30 \
                     --do_train=True \
                     --word_emb_dim=300 \
                     --max_seq_length=180 \
                     --seed=1234 \
                     





