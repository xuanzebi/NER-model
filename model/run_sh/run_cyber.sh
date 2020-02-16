export CUDA_VISIBLE_DEVICES=2
python ../main.py --use_scheduler=True \
                     --model_save_dir='/opt/hyp/NER/NER-model/saved_models/cyber_bilstm/' \
                     --tensorboard_dir='/opt/hyp/NER/NER-model/saved_models/cyber_bilstm/runs/' \
                     --batch_size=128 \
                     --optimizer='Adam' \
                     --momentum=0.9 \
                     --data_path='/opt/hyp/NER/NER-model/data/json_data' \
                     --dump_embedding=True \
                     --save_embed_path='/opt/hyp/NER/NER-model/data/Tencent_AILab_ChineseEmbedding.p' \
                     --pred_embed_path='/opt/hyp/NER/embedding/Tencent_AILab_ChineseEmbedding.txt' \
                     --load=True \
                     --num_train_epochs=100 \
                     --do_train=True \
                     --word_emb_dim=200 \

