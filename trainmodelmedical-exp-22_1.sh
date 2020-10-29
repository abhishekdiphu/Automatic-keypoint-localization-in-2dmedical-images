python trainmodelmdeical-exp22_1.py \
--path /home/aburagohain/long_leg_ruppertshofen/ruppertshofen_cleaned.iml \
--modelName train-model-19/supervised-medical-660-lr-00002/new_exp53_batchsize1 \
--config config.default_config \
--batch_size 1 \
--use_gpu \
--lr .00002 \
--print_every 50 \
--train_split 0.8505 \
--loss mse \
--optimizer_type Adam \
--epochs 60 \
--dataset  'medical' 



#all.iml -- model 16
#ruppertshofen_cleaned.iml - model 17
