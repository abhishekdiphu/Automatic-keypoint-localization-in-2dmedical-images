python trainmodelmedicaladverse_pose_small_dis-exp28.py \
--path handmedical \
--modelName train-model-35/adversarial_models_small_pose_pretrained/no_pretrain----- \
--config config.default_config \
--batch_size 5 \
--use_gpu \
--lr .0002 \
--print_every 150 \
--train_split 0.804 \
--loss mse \
--optimizer_type Adam \
--epochs 60 \
--dataset  'medical' 

