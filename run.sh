DATASET_NAME="RSTPReid"
CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
--name PPL \
--output_dir 'ITSELF' \
--dataset_name $DATASET_NAME \
--loss_names 'tal+cid' \
--num_epoch 15 \
--batch_size 64 \
--return_all \
--topk_type 'custom' \
--modify_k
# --only_global 
