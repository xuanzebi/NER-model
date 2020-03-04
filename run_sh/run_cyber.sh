export CUDA_VISIBLE_DEVICES=1
python ../model/lstm/main.py --use_scheduler=True \
                     --model_save_dir='/opt/hyp/NER/NER-model/saved_models/cyber/cyber_bilstm_muli_mtl/' \
                     --tensorboard_dir='/opt/hyp/NER/NER-model/saved_models/cyber/cyber_bilstm_muli_mtl/runs/' \
                     --batch_size=128 \
                     --optimizer='Adam' \
                     --momentum=0.9 \
                     --data_path='/opt/hyp/NER/NER-model/data/json_data' \
                     --dump_embedding=True \
                     --save_embed_path='/opt/hyp/NER/NER-model/data/embedding/Tencent_AILab_ChineseEmbedding_cyber.p' \
                     --pred_embed_path='/opt/hyp/NER/embedding/Tencent_AILab_ChineseEmbedding.txt' \
                     --load=True \
                     --num_train_epochs=150 \
                     --do_train=True \
                     --do_test=True \
                     --word_emb_dim=200 \
                     --seed=42 \
                     --max_seq_length=200 \
                     --dropout=0.2 \
                     --dropoutlstm=0.2 \
                     --use_crf=True \
                     --deal_long_short_data='cut' \
                     --model_classes='bilstm_mtl' \
                     --use_token_mtl=False \
                     --use_multi_token_mtl=True \
                     --use_bieos=True \
                     --use_packpad=False \
                     --save_best_model=True \
                     --freeze=False \
                     --use_fgm=False \
                     --msra_freeze=True \
                     --use_elmo=False \
                     --use_bert=False \
                     --FGM=False \





