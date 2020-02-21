export CUDA_VISIBLE_DEVICES=2
python ../model/lstm/main.py --use_scheduler=True \
                     --model_save_dir='/opt/hyp/NER/NER-model/saved_models/gpu2t/est/' \
                     --tensorboard_dir='/opt/hyp/NER/NER-model/saved_models/gpu2test/runs/' \
                     --batch_size=128 \
                     --optimizer='Adam' \
                     --momentum=0.9 \
                     --data_path='/opt/hyp/NER/NER-model/data/json_data' \
                     --dump_embedding=True \
                     --save_embed_path='/opt/hyp/NER/NER-model/data/embedding/Tencent_AILab_ChineseEmbedding_cyber.p' \
                     --pred_embed_path='/opt/hyp/NER/embedding/Tencent_AILab_ChineseEmbedding.txt' \
                     --load=True \
                     --num_train_epochs=50 \
                     --do_train=True \
                     --do_test=True \
                     --word_emb_dim=200 \
                     --seed=42 \
                     --max_seq_length=200 \
                     --dropout=0.2 \
                     --dropoutlstm=0.2 \
                     --use_crf=True \
                     --deal_long_short_data='cut' \
                     --model_classes='bilstm' \
                     --use_bieos=True \
                     --use_packpad=False \




