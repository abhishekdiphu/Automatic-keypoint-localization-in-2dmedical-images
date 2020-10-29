import torch
import numpy as np
import matplotlib.pyplot as plt
import hiwi
eps = 1e-8
import datasets.img as I

class Options(object):
    '''
    Presetting :
    
    Args :
        self.outputRes : output resolution of the model
        self.nStack : No of stack hour glass used in  the generator network
        
    
    '''
    def __init__(self, outputRes, nStack):
        self.outputRes = outputRes
        self.nStack = nStack
        

class PCK(object):
    """docstring for A.O Mader's metrics for lower leg datasets"""
    def __init__(self, opts):
        super(PCK, self).__init__()
        self.opts = opts
        self.input_resolution = 256
        self.output_resolution = 64
        

    def calc_dists(self, preds, target, normalize):
        '''
        Args:
            pred (multidimentional tensor) : the predicted heatmaps of the model.
            target (multidimentional tensor) : ground truth heatmaps of the x-rays images 
            normalize ( numpy-array) : to put  the pred and target in the same scale, not used .
        return:
            dists (numpy array (c, n)): n is  the batch size , c is the column , from  0 to 5, one each for a keypoints
            example:
            for batch of 4 images :
                                  joints(c)      
                          a_l a_r f_l f_r k_l k_r 
                          0    1   2   3  4    5
                          
                       0[[1    1   1   1  1    1],
           images(n)   1 [1    1   1   1  1    1], 
                       2 [1    1   1   1  1    1],
                       3 [1    1   1   1  1    1]]
            
        '''
        preds = (preds*(self.input_resolution/self.output_resolution)).astype(np.float32)
        #print(preds)
        target = (target*(self.input_resolution/self.output_resolution)).astype(np.float32)
        print(target)
        dists = np.zeros((preds.shape[1], preds.shape[0]))
        #mader = []
        #print(dists)
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 0 and target[n, c, 1] > 0:
                    normed_preds = preds[n, c, :] #/ normalize[n]
                    #print(normed_preds)
                    
                    normed_targets = target[n, c, :] #/ normalize[n]
                    #print(normed_targets)
                    
                    
                    
                    dists[c, n] = np.linalg.norm(normed_preds*np.array([4.1 , 4.1]) - normed_targets*np.array([4.1 , 4.1]))
                    
                    #dist  = hiwi.Evaluation(normed_targets ,normed_preds, 
                    #                               localized =np.less(dists[c, n], 10)  ,
                    #                               spacing = np.array([4.1 , 4.1]))
                    #mader.append(dist)
                    #print("{}".format(dist))

                   
                    
                    #print(dists[c , n])
               
                else:
                    #print(target[n, c, 0] , target[n, c, 1])
                    
                    dists[c, n] = -1
        
        #for i in mader:
        #    print("{}".format(i))
        #Avg = hiwi.AvgEvaluation(mader)
        #fig, ax = plt.subplots()
        #plot = Avg.plot_localizations(ax = ax , max_error= 4.1, max_error_mm= 1 )
        #plt.show()
        #plt.savefig("plot.png")
        
        #print("{}".format(Avg))
        
        return dists
     
        
    
    def dist_acc(self, dists, thr= 10):   #7.31  # 4.87 #2.44
         ''' Return percentage below threshold while ignoring values with a -1 
             10 mm = 2.44px
             20 mm = 4.87px
             30 mm = 7.31px
             4.1mm = 1px
             
             Args:
                 dists(numpy array): return value of calc_dists() -->dist[c , n]
                 thr(float or int) : threshold value in terms of px.
             Return:
                 float , Return percentage below  the threshold while ignoring values with a -1 
         
         '''
         dist_cal = np.not_equal(dists, -1)
        
         #print(" Total no of key-point present :" , dist_cal)
         
         num_dist_cal = dist_cal.sum()
         #print("Sum of present key-keypoints :", num_dist_cal ) 
         #print("trial :",np.less(dists, thr))   
         #print("2 :",np.less(dists[dist_cal], thr))
        
         if num_dist_cal > 0:
                
             #print("Accurately detected :",np.less(dists[dist_cal], thr).sum())   
             #print("dist_cal :",np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal)
                
             return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
         else:
             return -1
        
   



    def get_max_preds(self, batch_heatmaps):
        '''
        get predictions from score maps
        Args:
            heatmap (multi-dimentional array)s: numpy.ndarray([batch_size, num_joints, height, width])
        Return:
            pred (numpy array) : predictions of the models 
            
        '''
        assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        

        preds[:, :, 0] = (preds[:, :, 0]) % width 
        
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width) 
        #print(preds)
        
        
        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)
        #print("print pred_mask :", pred_mask)

        #print('before mask', preds)

        preds *= pred_mask
        #print('after mask', preds)
        return preds, maxvals

    def eval(self, pred, target, alpha=0.5):
        '''
        Calculate accuracy according to eulidian distance(true , predicted) < 10 mm , 20 mm , 30 mm,
        but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs',
        followed by individual accuracies
        
        Args:
            pred (multidimentional tensor) : the predicted heatmaps of the model.
            target (multidimentional tensor) : ground truth heatmaps of the x-rays images 
            alpha (float) : Not used 
        return:
            avg_acc (float) : Percentage of keypoints localized within the threshold.
            cnt (int) : total no of keypoints present
            new_pfake (numpy array): p_fake values as per the equations
            
        
        '''
        idx = list(range(6))
        print(len(idx))
        norm = 1.0
        if True:
             h = self.opts.outputRes
             w = self.opts.outputRes
             print("pred shape: ", pred.shape[0])   
             norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        dists = self.calc_dists(pred, target, norm)

        acc = np.zeros((len(idx) + 1))
        
        ##--------------------calculate minibatch of pfakes ------------------------------##
        #acc_for_pfake = np.zeros((len(idx) + 1))
        #new_pfake     = np.zeros((pred.shape[0], len(idx)))
        ##--------------------------------------------------------------------------------##
        
        avg_acc = 0.0
        cnt = 0
        
        
        p_fake =[]
        for i in range(len(idx)):
             acc[i + 1] = self.dist_acc(dists[idx[i]])
             if acc[i + 1] >= 0:
                 p_fake.append(int(acc[i + 1]))
             elif acc[i + 1] < 0:
                 p_fake.append(0)
             if acc[i + 1] >= 0:

                 avg_acc = avg_acc + acc[i + 1]
                 cnt += 1

             print('acc[%d] = %f' % (i + 1, acc[i + 1]))
        
        print(p_fake)
        
        avg_acc = 1.0 * avg_acc / cnt if cnt != 0 else 0
        if cnt != 0:
             acc[0] = avg_acc
        
        #-----------------------------------------------------------------------------#
        #for n in range(pred.shape[0]):
            
        #    for c in range(len(idx)):
                
        #        acc_for_pfake[i + 1] = self.dist_acc(dists[idx[c] ,n])
                
        #        if acc_for_pfake[i + 1] >= 0:
                    
        #            new_pfake[n ,c]= int(acc_for_pfake[i + 1])
                        
        #        elif acc_for_pfake[i + 1] < 0:
                    
        #             new_pfake[n ,c]= 1
        #print("===============p-fake===============")                
        #print(new_pfake)   
        #print("==============================================================================")
        return avg_acc,cnt,p_fake

    def StackedHourGlass(self, output, target, alpha=0.5):
        
        '''
        get the accuracy  by calculating eulclidian distance (pred , target) < 10 mm , 20 mm ,30mm
        
        Args:
            outputs (multidim tensor) : prediction heatmaps from the stack hour glass
            target  (multidim tensor) : ground truth heatmaps from the datasets.
            
        return :
            eval (function): self.eval() function is executed. 
        '''
        predictions = self.get_max_preds(output[self.opts.nStack-1].detach().cpu().numpy())
        
        comb_pred = np.sum(output[-1].detach().cpu().numpy()[0], axis=0)
        #print(comb_pred)
        
        #plt.imshow(comb_pred)
        #plt.colorbar()
        #plt.savefig('comb_hmap.png' , cmap = 'gist_gray' )  ## uncommented
        #plt.clf()
        #plt.imshow(np.sum(target.detach().cpu().numpy()[0], axis=0), cmap ='seismic')
        #plt.colorbar()
        #plt.savefig('gt_hmap.png') 
        #plt.clf()

        target = self.get_max_preds(target.cpu().numpy())
        
        
        
        return self.eval(predictions[0], target[0], alpha)

    