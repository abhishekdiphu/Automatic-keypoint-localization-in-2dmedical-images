
python trainmodeladversarial-pos-conf-exp24.py \
--path handmedical \
--modelName trainmodel-36/pose_conf_Adversarial_model/experiment21 \
--config config.default_config \
--batch_size 1 \
--use_gpu \
--lr .0002 \
--print_every 100 \
--train_split 0.804 \
--loss mse \
--optimizer_type Adam \
--epochs 50 \
--dataset  'medical' 


##@@@both conf and pose are being used @@####