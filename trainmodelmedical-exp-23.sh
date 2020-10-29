python trainmodelmdeicaladverse-exp23.py \
--path handmedical \
--modelName train-model-32/adversarial_models_pose_pretrained/experiment71/test/lr_0002_beta_180 \
--config config.default_config \
--batch_size 1 \
--use_gpu \
--lr .0002 \
--print_every 999999 \
--train_split 0.8505 \
--loss mse \
--optimizer_type Adam \
--epochs 60 \
--dataset  'medical' 



#lr_002_beta_180