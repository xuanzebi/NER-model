export CUDA_VISIBLE_DEVICES=1
python ../model/lstm/main.py --use_scheduler=True \
                     --model_save_dir='/opt/hyp/NER/NER-model/saved_models/msra_bilstm/' \
                     --tensorboard_dir='/opt/hyp/NER/NER-model/saved_models/msra_bilstm/runs/' \
                     --batch_size=128 \
                     --optimizer='Adam' \
                     --momentum=0.9 \
                     --data_path='/opt/hyp/NER/NER-model/data/other_data/MSRA/json_data' \
                     --dump_embedding=True \
                     --save_embed_path='/opt/hyp/NER/NER-model/data/embedding/Tencent_AILab_ChineseEmbedding_msra.p' \
                     --pred_embed_path='/opt/hyp/NER/embedding/Tencent_AILab_ChineseEmbedding.txt' \
                     --load=False \
                     --num_train_epochs=150 \
                     --use_dataParallel=False \
                     --do_train=True \
                     --word_emb_dim=200 \
                     --max_seq_length=156 \
                     --deal_long_short_data='cut' \






