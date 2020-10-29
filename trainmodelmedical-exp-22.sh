python trainmodelmdeical-exp22.py \
--path /home/aburagohain/long_leg_ruppertshofen/ruppertshofen_cleaned.iml \
--modelName train-model-19/supervised-medical-660-lr-002/entill-60 \
--config config.default_config \
--batch_size 1 \
--use_gpu \
--lr .002 \
--print_every 10000 \
--train_split 0.8505 \
--loss mse \
--optimizer_type Adam \
--epochs 58 \
--dataset  'medical' 



#all.iml -- model 16
#ruppertshofen_cleaned.iml - model 17


#epochs --- starts from 59