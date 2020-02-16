export CUDA_VISIBLE_DEVICES=0
python ../main.py --use_scheduler=False \
                     --model_save_dir='/opt/hyp/NER/NER-model/saved_models/bilstm_sgd/' \
                     --tensorboard_dir='/opt/hyp/NER/NER-model/saved_models/bilstm_sgd/runs/' \
                     --batch_size=64 \
                     --optimizer='SGD' \
                     --momentum=0 \
