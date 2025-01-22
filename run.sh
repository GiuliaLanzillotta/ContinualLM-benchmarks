max_samples=640000 
for idrandom in 0 
do    
 for pt_task in 0 1 2 3 4 5    
  do    
   CUDA_VISIBLE_DEVICES=1,7 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=localhost --master_port=10102 posttrain.py \
   --per_device_train_batch_size 64 --fp16 --max_seq_length 164 --max_samples ${max_samples}  --idrandom ${idrandom} --ntasks 6  --pt_task ${pt_task} --baseline 'ncl' --model_name_or_path distilroberta-base
 done 
done  
