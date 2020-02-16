export CUDA_VISIBLE_DEVICES=2
python ../main.py --use_scheduler=True \
                     --model_save_dir='/opt/hyp/NER/NER-model/saved_models/bilstm_resume/' \
                     --tensorboard_dir='/opt/hyp/NER/NER-model/saved_models/bilstm_resume/runs/' \
                     --batch_size=64 \
                     --optimizer='Adam' \
                     --momentum=0 \
                     --data_path='/opt/hyp/NER/NER-model/data/other_data/ResumeNER/json_data' \
                     --dump_embedding=True \
                     --save_embed_path='/opt/hyp/NER/NER-model/data/Tencent_AILab_ChineseEmbedding_resume.p' \
                     --load=False \

