"""
Specifies default parameters used for training in a dictionary
"""

config = {}

hourglass_params = {}
hourglass_params['num_reductions'] = 4
hourglass_params['num_residual_modules'] = 2

dataset = {}
dataset['num_joints'] = 6
#it was 16 in place of 14  #14 for lsp and #16 for mpii #please refer the no of joint cordinates present in the datatsets that you take

pose_discriminator = {}
# TODO: base case now; add 3 later for in_channels
pose_discriminator['in_channels'] = dataset['num_joints'] +3  #*2
pose_discriminator['num_channels'] = 128
# pose_discriminator['num_joints'] = dataset['num_joints']
pose_discriminator['num_residuals'] = 5 # originally it is 5


conf_discriminator = {}
conf_discriminator['in_channels'] = dataset['num_joints']*2



#--------------------------GENERATOR---------------------------------------------------#
generator = {}
generator['num_joints'] = dataset['num_joints']
generator['num_stacks'] = 8 
generator['hourglass_params'] = hourglass_params
generator['mid_channels'] = 512
generator['preprocessed_channels'] = 64

#------------------------DISCRIMINATOR-------------------------------------------------#



training = {}
training['gen_iters'] = 1
training['disc_iters'] = 1 
training['beta'] = 1.0 / 180
training['alpha']  = 1.0 / 220


config = {'dataset': dataset, 'generator': generator, 'discriminator': pose_discriminator, 
            'training': training}