python trainmodelmdeicaladverse_with_lr_schedule-exp29.py \
--path handmedical \
--modelName train-model-32/adversarial_models_pose_pretrained/experiment100_alpha_138_lr_0002_with_lr_schedule/by_factor_0_3 \
--config config.default_config \
--batch_size 1 \
--use_gpu \
--lr .0002 \
--print_every 150 \
--train_split 0.99999999999 \
--loss mse \
--optimizer_type Adam \
--epochs 60 \
--dataset  'medical' 



