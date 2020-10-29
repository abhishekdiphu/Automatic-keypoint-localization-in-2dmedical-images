

##### medical data #########


import argparse
import importlib
import random
import os
import shutil
import pickle
import json as json

import numpy as np
import matplotlib ## was not there # due to tkinter.TclError could not connect to the display
matplotlib.use('svg') ## was not there
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import cv2



from datasets.lsp import LSP
from datasets.mpii import MPII
from datasets.legmedical import HANDXRAY
from posenet import PoseNet as  Generator
#from generator import Generator
from discriminator import Discriminator
#from discriminator02 import DiscPoseNet as Discriminator2
#from discriminator02 import DiscConfNet as Discriminator3

from losses import gen_single_loss, disc_single_loss, get_loss_disc, get_loss_disc_bce_with_logits, disc_loss_pose
import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--modelName', required=True, help='name of model; name used to create folder to save model')
parser.add_argument('--config', help='path to file containing config dictionary; path in python module format')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in momentum optimizer')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training, testing')
parser.add_argument('--val_batch_size', type=int, default=1, help='batch size for validation')
parser.add_argument('--print_every', type=int, default=1000, help='frequency to print train loss, accuracy to terminal')
parser.add_argument('--save_every', type=int, default=350, help='frequency of saving the model')
parser.add_argument('--optimizer_type', default='SGD', help='type of optimizer to use (SGD or Adam or RMS-Prop)')
parser.add_argument('--use_gpu', action='store_true', help='whether to use gpu for training/testing')
parser.add_argument('--gpu_device', type=int, default=0, help='GPU device which needs to be used for computation')
parser.add_argument('--validation_sample_size', type=int, default=1, help='size of validation sample')
parser.add_argument('--validate_every', type=int, default=5, help='frequency of evaluating on validation set')

parser.add_argument('--path', \
    default='/datasets/lspet_dataset')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)
parser.add_argument('--train_split', type=float, default=0.804)
parser.add_argument('--heatmap_sigma', type=float, default=2)
parser.add_argument('--occlusion_sigma', type=float, default=2)
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--dataset', type=str, required=True, choices=['mpii', 'lsp', 'medical'])

args = parser.parse_args()

# initialize seed to constant for reproducibility
np.random.seed(58)
torch.manual_seed(58)
random.seed(58)

# create directory to store models if it doesn't exist
# if it exists, delete the contents in the directory
if (not os.path.exists(args.modelName)):
    os.makedirs(args.modelName)
else:
    shutil.rmtree(args.modelName)
    os.makedirs(args.modelName)

# for handling training over GPU
cpu_device = torch.device('cpu')
fast_device = torch.device('cpu')
if (args.use_gpu):
    fast_device = torch.device('cuda:' + str(args.gpu_device))

# config file storing hyperparameters
config = importlib.import_module(args.config).config

# Initializing the models
#generator_model = Generator(nstack = 8 , inp_dim = 256 , oup_dim = 6)

discriminator_model_pose = Discriminator(6 , config['discriminator']['num_channels'], config['dataset']['num_joints'], config['discriminator']['num_residuals'])

#discriminator_model_pose = Discriminator2(nstack = 1 , inp_dim = 256 , oup_dim = 6)

#discriminator_model_conf = Discriminator3(nstack = 1 , inp_dim = 256 , oup_dim = 6)


modelpath_g = torch.load('train-model-19/supervised-medical-660-lr-0002/experiment08/model_5_520.pt')
generator_model = modelpath_g['generator_model']
#print(generator_model)
#print(generator_model)
#print(discriminator_model)

####modelpath_d = torch.load('train-model-16-medical-pre-trainied_pose_conf/pretrained_conf_pose/pretrained_conf_pose_10_500.pt')

####discriminator_model_pose = modelpath_d['discriminator_model_pose']
####discriminator_model_conf = modelpath_d['discriminator_model_conf']

#print(generator_model)
#print(generator_model)
#print(discriminator_model)


#modelpath_d = torch.load('train-model-16-medical-adversarial/modeladversarial_40_500.pt')
#discriminator_model_pose = modelpath_d['discriminator_model_conf']
#print(generator_model)
#print(generator_model)
#print(discriminator_model)

for params in generator_model.parameters():
    params.requires_grad = False
#for params in discriminator_model_pose.parameters():
#    params.requires_grad = True
for params in discriminator_model_conf.parameters():
    params.requires_grad = True

# Use dataparallel
generator_model = nn.DataParallel(generator_model)
discriminator_model_conf = nn.DataParallel(discriminator_model_conf)
#discriminator_model_pose = nn.DataParallel(discriminator_model_pose)

# Datasets
if args.dataset == 'lsp':
    lsp_train_dataset = LSP(args)
    args.mode = 'val'
    lsp_val_dataset = LSP(args)
# medical    
if args.dataset == 'medical':
    lsp_train_dataset = HANDXRAY(args)
    args.mode = 'val'
    lsp_val_dataset = HANDXRAY(args)
# MPII
elif args.dataset == 'mpii':
    lsp_train_dataset = MPII('train')
    lsp_val_dataset = MPII('valid') ## MPII('val') was present originally

# Dataset and the Dataloade
train_loader = torch.utils.data.DataLoader(lsp_train_dataset, batch_size=args.batch_size, shuffle=True)
val_save_loader = torch.utils.data.DataLoader(lsp_val_dataset, batch_size=args.val_batch_size)
val_eval_loader = torch.utils.data.DataLoader(lsp_val_dataset, batch_size=args.val_batch_size, shuffle=True)
#train_eval = torch.utils.data.DataLoader(lsp_train_dataset, batch_size=args.val_batch_size, shuffle=True)


pck = metrics.PCK(metrics.Options(256, 8))




# Loading on GPU, if available
if (args.use_gpu):
    generator_model = generator_model.to(fast_device)
    discriminator_model_conf = discriminator_model_conf.to(fast_device)
#    discriminator_model_pose = discriminator_model_pose.to(fast_device)

# Cross entropy loss
#criterion = nn.CrossEntropyLoss()

# Setting the optimizer
if (args.optimizer_type == 'SGD'):
    optim_gen = optim.SGD(generator_model.parameters(), lr=args.lr, momentum=args.momentum)
    optim_disc = optim.SGD(discriminator_model.parameters(), lr=args.lr, momentum=args.momentum)

elif (args.optimizer_type == 'Adam'):
    optim_gen = optim.Adam(generator_model.parameters(), lr=args.lr) ## added the betas .originally not there 
    optim_disc_conf = optim.Adam(discriminator_model_conf.parameters(), lr=args.lr) ##added the betas.originally not there 
#    optim_disc_pose = optim.Adam(discriminator_model_pose.parameters(), lr=args.lr) ##added the betas.originally not there 
    
#----code added here inplementing rms-prob as an option for optimization--------#
elif (args.optimizer_type == 'RMS'):
    optim_gen =  optim.RMSprop(generator_model.parameters(), lr=args.lr)
    #optim_disc=  optim.RMSprop(discriminator_model.parameters(), lr=args.lr)
else:
    raise NotImplementedError
    
    
#----------------------------------------------------------------------------------#
#------------------------fine tuning the model-------------------------------------#
# pretrained model being Load model
#modelpath = torch.load('train_model04/model_228_5000.pt')
#generator_model = modelpath['generator_model']
#discriminator_model = modelpath['discriminator_model']
#optim_gen = modelpath['optim_gen']
#optim_disc = modelpath['optim_disc']
#print("pretrained model Loaded")
#-----------------------------------------------------------------------------------#
# The main training loop





gen_losses = [0.00,00.00]
disc_losses = []
disc_conf_losses = [] 
val_gen_losses = []
val_disc_losses = []

conf_real = []
conf_fake = []

pose_real =[]
pose_fake =[]


mean_accuracy =[]
mean_count    =[]

def evaluate_model(args, epoch, val_loader, fast_device, generator_model, discriminator_model): #val_loader
    os.makedirs(os.path.join(args.modelName, str(epoch)))
    gen_loss = 0.0
    disc_loss = 0.0
    disc_loss_pose = 0.0
    all_images = None
    all_outputs = []
    all_ground_truth = {}
    for i, data in enumerate(val_loader):
        print("index :",i)
        if i>=100: break  #---------------------------------------> validation samples 
        images = data['image']
        if (all_images is None):
            all_images = images.numpy()
        else:
            all_images = np.concatenate((all_images, images.numpy()), axis=0)
        ground_truth = {}
        ground_truth['heatmaps'] = data['heatmaps']
        ground_truth['occlusions'] = data['occlusions']
        if (len(all_ground_truth) == 0):
            for k, v in ground_truth.items():
                all_ground_truth[k] = ground_truth[k].numpy()
        else:
            for k, v in ground_truth.items():
                all_ground_truth[k] = np.concatenate((all_ground_truth[k], ground_truth[k].numpy()), axis=0)
        
        if (args.use_gpu):
            images = images.to(fast_device)
            ground_truth['heatmaps'] = ground_truth['heatmaps'].to(fast_device)
            ground_truth['occlusions'] = ground_truth['occlusions'].to(fast_device)

        with torch.no_grad():
            outputs = generator_model(images)
            cur_gen_loss_dic = gen_single_loss(ground_truth, outputs, discriminator_model, mode=args.loss)
 #           cur_disc_loss_dic = disc_single_loss(ground_truth, outputs, discriminator_model)

            cur_gen_loss = cur_gen_loss_dic['loss']
 #          cur_disc_loss = cur_disc_loss_dic['loss']

            gen_loss += cur_gen_loss
 #           disc_loss += cur_disc_loss

            if (len(all_outputs) == 0):
                for output in outputs:
                    all_outputs.append(output.to(cpu_device).numpy())
            else:
                for i in range(len(outputs)):
                    all_outputs[i] = np.concatenate((all_outputs[i], outputs[i].to(cpu_device).numpy()), axis=0)

    with open(os.path.join(args.modelName, str(epoch), 'validation_outputs.dat'), 'wb') as f:
        pickle.dump((all_images, all_ground_truth, all_outputs), f)
        
    return gen_loss #, disc_loss

val_pos = 0
for epoch in range(args.epochs):
    print('epoch:', epoch)

    if (epoch > 0 and epoch % args.save_every == 0):
        torch.save({'generator_model': generator_model,
                    'discriminator_model_conf': discriminator_model_conf,
                    'discriminator_model_pose': discriminator_model_pose,
#                    'criterion': criterion, 
                     'optim_gen': optim_gen , 
                    'optim_disc_conf': optim_disc_conf,
                     'optim_disc_pose': optim_disc_pose}, 
                    os.path.join(args.modelName, 'model_' + str(epoch) + '.pt'))
        val_gen_loss = evaluate_model(args, epoch, val_save_loader, fast_device, generator_model)
        val_gen_losses.append(val_gen_loss)
#       val_disc_losses.append(val_disc_loss)

    epoch_gen_loss = 0.0
#   epoch_disc_loss = 0.0
    print("length of training datasets: ",len(train_loader))
    print("length of validation datasets: ",len(val_eval_loader))
    for i, data in enumerate(train_loader):
#        optim_gen.zero_grad()
        optim_disc_conf.zero_grad()
#        optim_disc_pose.zero_grad()
        
        images = data['image']
        ground_truth = {}
        ground_truth['heatmaps'] = data['heatmaps']
        ground_truth['occlusions'] = data['occlusions']
        if (args.use_gpu):
            images = images.to(fast_device)
            ground_truth['heatmaps'] = ground_truth['heatmaps'].to(fast_device)
            ground_truth['occlusions'] = ground_truth['occlusions'].to(fast_device)

    ########################  Check and complete the code here ######## Forward pass and calculate losses here ################
        
        
        
        #if (i % (config['training']['gen_iters'] + config['training']['disc_iters']) < config['training']['gen_iters']):
            
        # --------------------------Generator training  by itself----------------------------------------------#
        print("epochs{}-------------Generator training  by itself--------------------------------".format(epoch))
            
        #optim_gen.zero_grad()
        #outputs = generator_model(images) 
        #print("the number of intermidiate loss in the generator  : ", len(outputs))
        
        #labels = ground_truth['heatmaps']



        #cur_gen_loss_dic = gen_single_loss(ground_truth, outputs, discriminator_model_conf, mode=args.loss)
        #cur_disc_loss_dic_conf = 0.0
        #cur_disc_loss_dic_pose = 0.0
        #for output in outputs:
        #    cur_disc_loss_dic_conf += get_loss_disc_bce_with_logits(output, discriminator_model_conf, real=True, 
        #                                                            ground_truth = ground_truth['heatmaps'])
        #    cur_disc_loss_dic_pose += get_loss_disc_bce_with_logits(torch.cat([output, images], 1), discriminator_model_pose, real=True, \
        #                                                            ground_truth=ground_truth['heatmaps'])

        #cur_disc_loss_dic_conf= cur_disc_loss_dic_conf/len(outputs)
        #cur_disc_loss_dic_pose= cur_disc_loss_dic_pose/len(outputs)

        #cur_gen_loss = cur_gen_loss_dic['loss']
        #print("generator real loss cur_disc_loss_dic_conf", cur_disc_loss_dic_conf)
        #print("generator real loss cur_disc_loss_dic_pose", cur_disc_loss_dic_pose)
        #cur_disc_loss = 1/220*(cur_disc_loss_dic_conf) + 1/180*(cur_disc_loss_dic_pose)

        #loss = cur_gen_loss # + config['training']['alpha'] * cur_disc_loss
        #print("single generator loss {}".format(loss))
        #loss.backward()

        #optim_gen.step()
            
            
        #else:
        #---------------------------confidence Discriminator training-------------------------------------------#
        print("---------------------confidence Discriminator training-------------------------------------------")
        optim_disc_conf.zero_grad()

        outputs_conf = generator_model(images)
        pck_out_conf = outputs_conf
        outputs_conf = [nn.Upsample(scale_factor=256 / 64, mode='nearest')(outputs_conf[-1].detach())]
        _ , _, p_fake = pck.StackedHourGlass(pck_out_conf, ground_truth['heatmaps'])
        
        
        cur_disc_loss_conf = disc_single_loss(ground_truth, outputs_conf, discriminator_model_conf, detach=False,p_fake = p_fake )

        #cur_gen_loss_dic = gen_single_loss(ground_truth, outputs, discriminator_model_conf, mode=args.loss)
        #cur_disc_loss_dic = disc_single_loss(ground_truth, outputs, discriminator_model_conf, detach=False)

        #cur_gen_loss = cur_gen_loss_dic['loss']
        loss_conf_fake =     cur_disc_loss_conf['pose_disc_real']
        loss_conf_real =     cur_disc_loss_conf['pose_disc_fake']
        cur_disc_loss_conf = cur_disc_loss_conf['loss']
        

        loss_conf = cur_disc_loss_conf
#        print("conf_disc loss {}".format(loss_conf))


        loss_conf.backward()


        optim_disc_conf.step()


#        #----------------------------pose Discriminator training -----------------------------------------------#
#
#        print("---------------------pose Discriminator training-------------------------------------------------")
#        optim_disc_pose.zero_grad()

#        outputs = generator_model(images)
#        pck_out = outputs
#        outputs = [nn.Upsample(scale_factor=256 / 64, mode='nearest')(outputs[-1].detach())]
        
#        _ , _, c_fake = pck.StackedHourGlass(pck_out, ground_truth['heatmaps'])


         #cur_gen_loss_dic = gen_single_loss(ground_truth, outputs, discriminator_model, mode=args.loss)
#        cur_disc_loss_dic = disc_loss_pose(ground_truth, outputs, images , discriminator_model_pose, detach=False, p_fake = c_fake )

        #cur_gen_loss = cur_gen_loss_dic['loss']
#        cur_disc_loss = cur_disc_loss_dic['loss']
#        loss_pose_fake = cur_disc_loss_dic['pose_disc_fake']
#        loss_pose_real = cur_disc_loss_dic['pose_disc_real']

#        loss_pose = cur_disc_loss
        #print("pose conf loss {}".format(loss_pose))



#        loss_pose.backward()


#        optim_disc_pose.step()


        #-----------------------------Adversarial generator training ---------------------------------------------#
        print("----------------------Adversarial generator training ------------------------------------------------")
        #optim_gen.zero_grad()
        
        #outputs = generator_model(images)
        #pck_out = outputs 
        
        #outputs_disc = [nn.Upsample(scale_factor=256 / 64, mode='nearest')(outputs[-1].detach())]
        
        #print("outputs  for the generator :",len(outputs))
#        outputs = [outputs[-1]]
#        print("outputs  for the discriminators :",len(outputs))
        #_ , _, p_fake = pck.StackedHourGlass(pck_out, ground_truth['heatmaps'])




        #cur_gen_loss_dic = gen_single_loss(ground_truth, outputs, None, mode=args.loss)
        
        #cur_gen_loss_dic = gen_single_loss(ground_truth, outputs, mode=args.loss)
        #cur_disc_loss_dic_conf = 0.0
        
        #cur_disc_loss_dic_pose = 0.0
        #for output in outputs_disc:
        #    cur_disc_loss_dic_conf += get_loss_disc_bce_with_logits(output, discriminator_model_conf, real=False, 
        #                                                            ground_truth =ground_truth['heatmaps'] , p_fake = p_fake )
        #    cur_disc_loss_dic_pose += get_loss_disc_bce_with_logits(torch.cat([output, images], 1), discriminator_model_pose, real=False,\
        #                                                           ground_truth =ground_truth['heatmaps'],p_fake = p_fake)



        #cur_gen_loss = cur_gen_loss_dic['loss']
        
        #print("generator real loss cur_disc_loss_dic_pose", cur_disc_loss_dic_pose)
        
        #cur_disc_loss =(cur_disc_loss_dic_pose)/180.  

        #loss = cur_gen_loss     + (cur_disc_loss_dic_pose)/220. +  cur_disc_loss_dic_conf/180.
        
        #print("adversarial loss {}".format(loss))
        
        #loss.backward()

        #optim_gen.step()
            
            
        # for  memory error , we have to use item()  and convert them to float no from tensors .   
#----------------------------------------------------------appending losses ---------------------------------------------------------#
        #cur_gen_loss = cur_gen_loss.item()
        cur_disc_conf_loss =cur_disc_loss_conf.item() 
#        cur_disc_loss = cur_disc_loss.item()
        #gen_losses.append(float(cur_gen_loss))
#        disc_losses.append(float(cur_disc_loss))
        disc_conf_losses.append(float(cur_disc_conf_loss))
        
        #loss_conf_fake = loss_conf_fake.item()
        #loss_conf_real = loss_conf_real.item()tr

        #loss_pose_fake =loss_pose_fake.item()
        #loss_pose_real =loss_pose_real.item()
        
        conf_fake.append(float(loss_conf_fake)) 
        conf_real.append(float(loss_conf_real))
#        pose_real.append(float(loss_pose_fake)) 
#        pose_fake.append(float(loss_pose_real))
        
        
        
        with open(os.path.join(args.modelName, 'stats.bin'), 'wb') as f:
            pickle.dump((gen_losses, disc_conf_losses, conf_fake, conf_real), f)                  ##disc_losses,, val_disc_losses, val_gen_losses
        ##print("Train iter: %d, generator loss : %f" % (i ,gen_losses[-1]))    
        #print("Train iter: %d, generator loss : %f, discriminator loss : %f" % (i ,gen_losses[-1], disc_losses[-1]))
        print("Train iter: %d, generator loss : %f, pose discriminator loss : %f" % (i ,gen_losses[-1], disc_conf_losses[-1]))

        if i % args.print_every == 0:
            print("Train iter: %d, generator loss : %f, pose discriminator loss : %f" % (i ,gen_losses[-1], disc_conf_losses[-1]))

            plt.clf()
            plt.imshow(np.sum(outputs_conf[-1].detach().cpu().numpy()[0][0 :6], axis=0))
            plt.savefig('trainingImages/train_output_exp27-pretrained.png')
            
            
            plt.clf()
            plt.imshow(np.sum(ground_truth['heatmaps'].detach().cpu().numpy()[0], axis=0))
            plt.savefig('trainingImages/train_gt_exp27-pretrained.png')

            plt.clf()
            print("image shape_exp23-pretrained: ", images.shape)
            plt.imshow((images.detach().cpu().numpy()[0].transpose(1, 2, 0) * 128 + 128).astype(np.uint8)) ## originally it was there , replaced by below line
            #plt.imshow((images.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255)
            plt.savefig('trainingImages/train_img_exp27.png')
        # Save model
        ######################################################################################################################
        if i > 0 and i % 500 == 0:
            # Saving the model and the losses
            torch.save({'generator_model': generator_model, 
                        'discriminator_model_conf': discriminator_model_conf,
                        #'discriminator_model_pose': discriminator_model_pose,
                        #'criterion': criterion, 
#                        'optim_gen': optim_gen, 
                        'optim_disc_conf': optim_disc_conf},
#                        'optim_disc_pose': optim_disc_pose }, \
                         os.path.join(args.modelName, 'pretrained_conf_{}_{}.pt'.format(epoch, i)))

        
        mean_eval_avg_acc, mean_eval_cnt = 0.0, 0.0
        if i % args.validate_every == 0:
            for j, data in enumerate(val_eval_loader):
                #if j>=100 : break #---New line 
                if (j == args.validation_sample_size):
                    break
                images = data['image']
        
                ground_truth = {}
                ground_truth['heatmaps'] = data['heatmaps']
                ground_truth['occlusions'] = data['occlusions']
        
                if (args.use_gpu):
                    images = images.to(fast_device)
                    ground_truth['heatmaps'] = ground_truth['heatmaps'].to(fast_device)
                    ground_truth['occlusions'] = ground_truth['occlusions'].to(fast_device)

                with torch.no_grad():
                    outputs = generator_model(images)
                    outputs[-1] = outputs[-1][:, : config['dataset']['num_joints']]
                    eval_avg_acc, eval_cnt, p_fake = pck.StackedHourGlass(outputs, ground_truth['heatmaps'])
                    mean_eval_avg_acc += eval_avg_acc
                    mean_eval_cnt += eval_cnt
#------------------append the calulated accuracy  to a list to visualize the the pck values over each ititation-------------#            
                    #print('type of ', type(mean_eval_avg_acc))
                    #print('type of ', type(mean_eval_cnt))
            mean_accuracy.append(float(mean_eval_avg_acc.item()))
            mean_count.append(mean_eval_cnt)
            
            print("Validation avg acc: %f, eval cnt: %f" % (mean_eval_avg_acc, mean_eval_cnt))
            with open(os.path.join(args.modelName, 'pckstats.bin'), 'wb') as p:
                 pickle.dump((mean_accuracy, mean_count), p)
#--------------------------------------------------------------------------------------------------------------------------#

  
    
#    with open(os.path.join(args.modelName, 'stats.bin'), 'wb') as f:
#            pickle.dump((gen_losses), f)




# Plotting the loss function
plt.plot(loss.detach().cpu().numpy()) # detach().cpu() was not there
plt.savefig(os.path.join(args.modelName, 'loss_graph.pdf')) ##orginally it was there .the line below is being added , due to error 
#plt.savefig(os.path.join(args.modelName, 'loss_graph.png'))
plt.clf()