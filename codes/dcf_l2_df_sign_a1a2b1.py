#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import argparse
import torch
import torch.utils.data.dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import *
from data_processing_smi_sign_a1a2b1 import *
import numpy as np 
import pandas as pd 
import os
import sys
import time
import datetime as dtym
from datetime import datetime
import glob
from pathlib import Path
import gc
import config as cfg
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from torch.nn.parameter import Parameter
from collections import OrderedDict


# In[2]:


torch.set_num_threads(2)
print('threads:',torch.get_num_threads())


# ## DeConFuse Network

# In[3]:


def calOutShape(input_shape, ksize1 = 3,ksize2 = 3, stride = 1,
                maxpool1 = False, maxpool2 = False, mpl_ksize = 2):
    mpl_stride = 2
    pad = ksize1//2
    dim1 = int((input_shape[2] - ksize1 + 2 * pad)/stride) + 1
    if maxpool1 == True:
        dim1 = (dim1 - mpl_ksize)//mpl_stride + 1
    pad = ksize2//2
    dim1 = int((dim1 - ksize2 + 2 * pad)/stride) + 1
    if maxpool2 == True:
        dim1 = (dim1 - mpl_ksize)//mpl_stride + 1
    return dim1


# In[4]:


class Transform(nn.Module):
    
    def __init__(self,input_shape, out_planes1 = 8, out_planes2 = 16, ksize1 = 3,ksize2 = 3,
                 maxpool1 = False, maxpool2 = False,
                 mpl_ksize = 2, num_channels = 2, activFunc = 'selu', atom_ratio = 0.5):
        
        super(Transform, self).__init__()
        self.ksize1 = ksize1
        self.ksize2 = ksize2
        self.mpl_ksize = mpl_ksize
        self.out_planes1 = out_planes1
        self.out_planes2 = out_planes2
        self.init_T(input_shape)
        self.maxpool1 = maxpool1
        self.maxpool2 = maxpool2
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.activFunc = activFunc
        self.activation = activation_funcs[self.activFunc]
        self.i = 1
        self.atom_ratio = atom_ratio #0.5
        self.init_X()
        
    
    def init_T(self,input_shape):
        
        conv = nn.Conv1d(input_shape[1], out_channels = self.out_planes1, kernel_size = self.ksize1, 
                         stride = 1, bias = True)
        self.T1 = conv._parameters['weight']
        conv = nn.Conv1d(in_channels = self.out_planes1, out_channels = self.out_planes2, kernel_size = self.ksize2, 
                         stride = 1, bias = True)
        self.T2 = conv._parameters['weight']
        
        
    def init_X(self):
        
        dim1 = calOutShape(self.input_shape,self.ksize1,self.ksize2,stride = 1,maxpool1 = self.maxpool1, 
                           maxpool2 = self.maxpool2, mpl_ksize = self.mpl_ksize)
        
        X_shape = [self.input_shape[0],self.out_planes2,dim1]
        self.X  = nn.Parameter(torch.randn(X_shape), requires_grad=True)
        
        self.num_features = self.out_planes2 * dim1 
        self.num_atoms = int(self.num_features*self.atom_ratio * self.num_channels) 
        
        T_shape = [self.num_atoms,self.num_features]
        self.T = nn.Parameter(torch.randn(T_shape), requires_grad=True)
        
        
    def forward(self, x):
        x = F.conv1d(x, weight = self.T1, stride = 1,padding = self.ksize1//2)
        if self.maxpool1:
            x = F.max_pool1d(x, self.mpl_ksize)
        x = self.activation(x)
        x = F.conv1d(x, weight = self.T2, stride = 1,padding = self.ksize2//2)
        if self.maxpool2:
            x = F.max_pool1d(x, self.mpl_ksize)
        
        y = torch.mm(self.T,x.view(x.shape[0],-1).t())
        return x, y
        
          
    def get_params(self):
        return self.T1, self.T2, self.X, self.T
    
    
    def X_step(self):
        self.X.data = torch.clamp(self.X.data, min=0)


    def Z_step(self):
        self.Z.data = torch.clamp(self.Z.data, min=0)
        
        
    def get_TZ_Dims(self):
        return self.num_features,self.num_atoms, self.input_shape[0]
    

class Network(nn.Module): 
    def __init__(self, inputs_shape, out_planes1 = 8, out_planes2 = 16, ksize1 = 3, 
                 ksize2 = 3, maxpool1 = False, maxpool2 = False,
                 mpl_ksize = 2, num_classes = 6, num_channels = 2, activFunc = 'selu', 
                 atom_ratio = 0.5, of_fc = 2):
        
        super(Network, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # creating 2 channels networks 
        self.channels_trnsfrm = nn.ModuleList()
        
        for ch in range(self.num_channels):
#             print(ch)
            
            t = Transform(inputs_shape, out_planes1 = out_planes1, out_planes2 = out_planes2, ksize1 = ksize1,
                                    ksize2 = ksize2, maxpool1 = maxpool1, maxpool2 = maxpool2, 
                                    mpl_ksize = mpl_ksize, num_channels = self.num_channels,
                                    activFunc = activFunc, atom_ratio = atom_ratio)
                
            self.channels_trnsfrm.append(t)
            
        self.num_classes = num_classes
        self.num_features, self.num_atoms, self.input_shape = self.channels_trnsfrm[0].get_TZ_Dims()
        Z_shape = [self.num_atoms, self.input_shape]
        self.Z = nn.Parameter(torch.randn(Z_shape), requires_grad = True)
        self.pred_list = []
        self.tl_features = []
        self.init_TX()
        self.out_feats = self.num_atoms//of_fc #v2
        self.fc = nn.Linear(self.num_atoms, self.out_feats)
        
        
    def init_TX(self):
        self.T1  = []
        self.T2 = []
        self.T3 = []
        self.T = []
        self.X_list = []
        for num in range(self.num_channels): 
            T1, T2, X, T = self.channels_trnsfrm[num].get_params()
#             print('X shape : init_TX', X.shape)
            self.T1.append(T1)
            self.T2.append(T2)
            self.X_list.append(X)
            self.T.append(T)
        self.T1 = torch.stack(self.T1,1)
        self.T2 = torch.stack(self.T2,1)
        #self.X_list = torch.stack(self.X_list,1) 
        self.T = torch.stack(self.T,1) 
        
        
    def forward(self,x):
        
        samples, channels, in_channel,features = x.shape
        self.pred_list = []
        self.outp = []
        
        for num in range(self.num_channels):
            temp_out, temp_outp = self.channels_trnsfrm[num](x[:,num,:])
            temp_out = temp_out.view(temp_out.size(0),-1)
            self.pred_list.append(temp_out)
            self.outp.append(temp_outp)
            
        
        i = 0
        for num in range(self.num_channels):
            if i == 0:
                self.tl_features = self.outp[num] 
            i += 1
            self.tl_features += self.outp[num]
            
        out1 = self.fc(self.tl_features.t())
        return out1

    
    
    def X_step(self):
        
        for num in range(self.num_channels): 
            self.channels_trnsfrm[num].X_step()
        
        
    def Z_step(self):
        
        self.Z.data = torch.clamp(self.Z.data, min = 0)
        
    
    def conv_loss_distance(self):
        
        self.init_TX()
        
        loss = 0.0
        for i in range(len(self.pred_list)): 
            predictions = self.pred_list[i].view(self.pred_list[i].size(0), -1)
            
            X = self.X_list[i].view(self.X_list[i].size(0),-1)
            
            Y = predictions - X[0:predictions.shape[0]]
            loss += Y.pow(2).mean()
            
        return loss
    
        
    def conv_loss_logdet(self):

        loss = 0.0
        
        for T in self.T1:
            T = T.view(T.shape[0],-1)
            U, s, V = torch.svd(T)
            loss += -s.log().sum()
            
        for T in self.T2:
            T = T.view(T.shape[0],-1)
            U, s, V = torch.svd(T)
            loss += -s.log().sum()
            
            
        return loss
        
        
    def conv_loss_frobenius(self):
        
        loss = 0.0
        
        for T in self.T1:
            loss += T.pow(2).sum()
        
        for T in self.T2:
            loss += T.pow(2).sum()
            
        return loss
    

    def loss_distance(self):

        loss = 0.0
        
        predictions = self.tl_features
        Y = predictions - self.Z[:,0:predictions.shape[1]] #Z
        loss += Y.pow(2).mean()    
        
        return loss
        
        
    def loss_logdet(self):
        
        loss = 0.0
        
        T = self.T.view(self.T.shape[0],-1)
        U, s, V = torch.svd(T)
        loss = -s.log().sum()
        
        return loss
        
        
    def loss_frobenius(self):
        
        loss = 0.0       
        loss = self.T.pow(2).sum()
        
        return loss


    def computeLoss(self, lam, mu, batch_idx, batch_size):
        
        loss1 = self.conv_loss_distance(batch_idx, batch_size)
        loss2 = self.conv_loss_frobenius() * mu
        loss3 = self.conv_loss_logdet() * lam
        loss4 = self.loss_distance(batch_idx, batch_size)
        loss5 = self.loss_frobenius() * mu
        loss6 = self.loss_logdet() * lam
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        
        return loss

    
    def getTZ(self):
        
        return self.T.view(self.T.shape[0],-1), self.Z
    
    
    def getX(self):
        
        return self.X_list
    
    
    def get_out_feature_size(self):
        
        return self.out_feats


# ## Decision Tree Network 

# In[4]:


class Tree(nn.Module):
    def __init__(self,depth,n_in_feature,used_feature_rate,n_class,jointly_training = True):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class
        self.jointly_training = jointly_training

        # used features in this tree
        n_used_feature = int(n_in_feature*used_feature_rate)
        onehot = np.eye(n_in_feature)
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor),requires_grad=False)
        # leaf label distribution
        if jointly_training:
            self.pi = np.random.rand(self.n_leaf,n_class)
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor),requires_grad=True)
        else:
            self.pi = np.ones((self.n_leaf, n_class)) / n_class
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor), requires_grad=False)

        # decision
        self.decision = nn.Sequential(OrderedDict([
                        ('linear1',nn.Linear(n_used_feature,self.n_leaf)),
                        ('sigmoid', nn.Sigmoid()),
                        ]))

    def forward(self,x):
        """
        :param x(Variable): [batch_size,n_features]
        :return: route probability (Variable): [batch_size,n_leaf]
        """

        feats = torch.mm(x,self.feature_mask) # ->[batch_size,n_used_feature]
        decision = self.decision(feats) # ->[batch_size,n_leaf]

        decision = torch.unsqueeze(decision,dim=2)
        decision_comp = 1-decision
        decision = torch.cat((decision,decision_comp),dim=2) # -> [batch_size,n_leaf,2]

        # compute route probability
        # note: we do not use decision[:,0]
        batch_size = x.size()[0]
        _mu = Variable(x.data.new(batch_size,1,1).fill_(1.))
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            _mu = _mu.view(batch_size,-1,1).repeat(1,1,2)
            _decision = decision[:, begin_idx:end_idx, :]  # -> [batch_size,2**n_layer,2]
            _mu = _mu*_decision # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer+1)

        mu = _mu.view(batch_size,self.n_leaf)

        return mu

    def get_pi(self):
        if self.jointly_training:
            return F.softmax(self.pi,dim =-1)
        else:
            return self.pi

    def cal_prob(self,mu,pi):
        """
        :param mu [batch_size,n_leaf]
        :param pi [n_leaf,n_class]
        :return: label probability [batch_size,n_class]
        """
        p = torch.mm(mu,pi)
        return p


    def update_pi(self,new_pi):
        self.pi.data=new_pi


class Forest(nn.Module):
    def __init__(self,n_tree,tree_depth,n_in_feature,tree_feature_rate,n_class,jointly_training):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree  = n_tree
        for _ in range(n_tree):
#             print('cmn')
            tree = Tree(tree_depth,n_in_feature,tree_feature_rate,n_class,jointly_training)
            self.trees.append(tree)

    def forward(self,x):
        probs = []
        for tree in self.trees:
            mu = tree(x)
            p=tree.cal_prob(mu,tree.get_pi())
            probs.append(p.unsqueeze(2))
        probs = torch.cat(probs,dim=2)
        prob = torch.sum(probs,dim=2)/self.n_tree

        return prob




class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        super(NeuralDecisionForest, self).__init__()
        self.feature_layer = feature_layer
        self.forest = forest

    def forward(self, x):
        out1 = self.feature_layer(x)
        out = out1.view(x.size()[0],-1)
        out = self.forest(out)
        return out1, out


# In[5]:


def prepare_model(dcf_config, forest_config, n_class = 2, jointly_training = True):
    epochs, batch_size, lamda_mean, lr, ks_f, wd, mpl_ks, maxpools, activFunc, atom_ratio, amsFlg, xstep_flg, zstep_flg, of_fc = dcf_config
    out_pl1, ks1, out_pl2, ks2 = ks_f
    maxpl1, maxpl2 = maxpools
    lamda, mean = lamda_mean
    feat_layer = Network(inputs_shape = (batch_size, 1, 384), out_planes1 = out_pl1, out_planes2 = out_pl2, ksize1 = ks1, 
                 ksize2 = ks2, maxpool1 = maxpl1, maxpool2 = maxpl2, 
                 mpl_ksize = mpl_ks, num_classes = n_class, num_channels = 2, 
                 activFunc = activFunc, atom_ratio = atom_ratio, of_fc = of_fc)
     
    n_tree, tree_depth, tree_feat_rate = forest_config
    forest = Forest(n_tree = n_tree, tree_depth = tree_depth, 
                    n_in_feature = feat_layer.get_out_feature_size(),tree_feature_rate = tree_feat_rate,
                    n_class = n_class, jointly_training = jointly_training)
    model = NeuralDecisionForest(feat_layer, forest)

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    return model



def dcf_model_configs():
    n_epochs = [75]
    n_batch = [4096]
    lam_mu_list = [(0.0001,0.00001)]
    learning_rate_list = [0.01]
    ks_filters = [(4,3,8,3)]
    wd_list = [1e-5]
    mpl_ks_list = [2]
    maxpool_list = [(True, True)]
    activations = ['selu']
    atom_ratio = [0.75]
    n_tree = [90]
    tree_depth = [10]
    tree_feature_rate = [0.5]
    amsFlg = [True]
    xstep_flg = [True]
    zstep_flg = [False]
    fc_out_feat_factor = [2]
    # create configs
    configs = list()
    for i in n_epochs:
        for j in n_batch:
            for m in lam_mu_list:
                for l in learning_rate_list:
                    for ks_f in ks_filters:
                        for wd in wd_list:
                            for mpl_ks in mpl_ks_list:
                                for maxpools in maxpool_list: 
                                    for activ_func in activations:
                                        for atm in atom_ratio:
                                            for ams in amsFlg:
                                                for x in xstep_flg:
                                                    for z in zstep_flg:
                                                        for of_fc in fc_out_feat_factor: 
                                                            for nt in n_tree:
                                                                for td in tree_depth:
                                                                    for feat_rate in tree_feature_rate:
                                                                        cfg = [i, j, m, l, ks_f, wd, mpl_ks, maxpools, 
                                                                               activ_func, atm, ams, x, z, of_fc, 
                                                                               nt, td, feat_rate]
                                                                        configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs



# In[6]:


def train_on_batch(X_train, Y_train, dcf_config, forest_config):
    
    epochs, batch_size, lamda_mean, lr, ks_f, wd, mpl_ks, maxpools, activFunc, atom_ratio, amsFlg, xstep_flg, zstep_flg, of_fc = dcf_config
    out_pl1, ks1, out_pl2, ks2 = ks_f
    maxpl1, maxpl2 = maxpools
    lamda, mean = lamda_mean
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = prepare_model(dcf_config, forest_config, n_class = 2)
    train_loader = DataLoader(DrugsData(X_train, Y_train), batch_size = batch_size, num_workers = 0, shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, 
                                 betas = (0.9, 0.999), eps = 1e-08, weight_decay = wd, amsgrad = amsFlg)
    report_every = 1#10
    t0 = time.time()
    train_accs = []
    train_losses = []
    ctl_losses = []
    class_losses = []
    for epoch in range(1, epochs + 1):
        # Update \Theta
        model.train()
        correct = 0
        total = 0 
        running_loss = 0.0
        ctl_run_loss = 0.0
        class_run_loss = 0.0
        ypred = []
        scores = []
        ytrue = []
        t00 = time.time()
        for batch_idx, (X1, X2, target) in enumerate(train_loader):
            input1, input2, labels = map(lambda x: Variable(x), [X1, X2, target])
            target = target.long()
            data = torch.stack([input1,input2],1)
            optimizer.zero_grad()
            out1, output = model(data)
            pred = torch.argmax(output.data, dim = 1)
            pred = pred.long()
            ypred.extend(pred.tolist())
            scores.extend(output.tolist())
            ytrue.extend(target.tolist())
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            loss1 = model.feature_layer.computeLoss(lamda, mean, batch_idx, batch_size)
            loss2 = F.nll_loss(torch.log(output), target)
            loss =  loss1 + loss2
            loss.backward()
            running_loss += loss.item()
            ctl_run_loss += loss1.item()
            class_run_loss += loss2.item()
            optimizer.step()
            if xstep_flg:
                model.feature_layer.X_step()
            if zstep_flg:
                model.feature_layer.Z_step()
            if batch_idx % report_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),loss.item()))
 
        train_accuracy = correct.item()/len(train_loader.dataset)
        train_accs.append(train_accuracy)
        train_losses.append(running_loss/(batch_idx + 1))
        ctl_losses.append(ctl_run_loss/(batch_idx + 1))
        class_losses.append(class_run_loss/(batch_idx + 1))
        t01 = time.time()
        print('Time taken for one epoch : ', str(dtym.timedelta(seconds = t01 - t00)))
        print('*'*100)
        print('Train Epoch: {} \tLoss: {:.4f}, Accuracy: {:.4f}'.format(
                    epoch, running_loss/(batch_idx + 1), (correct.item()/len(train_loader.dataset))))
    scores = np.asarray(scores)
    torch.save({
            'epochs': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 
            'model': model
            },model_path) 
    tr_precision, tr_recall, tr_f1_score,_ = precision_recall_fscore_support(ytrue, ypred, average = 'weighted', 
                                                                             labels = [0,1])
    train_auc = roc_auc_score(ytrue, scores[:,1], average = 'weighted', labels = [0,1])
    print('Train Accuracy:', accuracy_score(ytrue, ypred)*100)
    print('Train F1 Score:', tr_f1_score)
    print('Train AUC:', train_auc)
    t1 = time.time()
    scores_dict = {}
    scores_dict['feat_types'] = str_fl_nm  
    scores_dict['seed'] = seed
    scores_dict['wd'] = wd
    scores_dict['lr'] = lr
    scores_dict['epochs'] = epochs
    scores_dict['lamda'] = lamda
    scores_dict['mean'] = mean
    scores_dict['ks1'] = ks1
    scores_dict['ks2'] = ks2
    scores_dict['out_pl1'] = out_pl1
    scores_dict['out_pl2'] = out_pl2
    scores_dict['maxpl1'] = maxpl1
    scores_dict['maxpl2'] = maxpl2
    scores_dict['amsFlg'] = amsFlg
    scores_dict['zstep_flg'] = zstep_flg
    scores_dict['xstep_flg'] = xstep_flg
    scores_dict['activFunc'] = activFunc
    scores_dict['loss_func'] = loss_func
    scores_dict['batch_size'] = batch_size
    scores_dict['train_loss'] = train_losses
    scores_dict['ctl_loss'] = ctl_losses
    scores_dict['class_loss'] = class_losses
    scores_dict['train_accs'] = train_accs
    scores_dict['train_auc'] = train_auc
    scores_dict['train_f1'] = tr_f1_score
    scores_dict['train_pr'] = tr_precision
    scores_dict['train_rec'] = tr_recall
    scores_dict['train_accuracy'] = accuracy_score(ytrue, ypred)
    scores_dict['train_scores'] = scores  
    print('Time taken for entire training : ', str(dtym.timedelta(seconds = t1 - t0)))
    print('*'*50)
    return scores_dict
              
              
                
def test(seed, X_test, Y_test, path, scores_dict, config_id):
     # Eval
    results_file_name = "../Results/Dcf_Dtf_l2_smi_sign" + str_fl_nm + "/res_dcf_dtf2_" + param_path + ".csv"
    pred_file_name = "../Results/Dcf_Dtf_l2_smi_sign" + str_fl_nm + "/pred_dcf_dtf2_" + param_path + "_cfg_" + str(config_id) + ".csv"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    test_loader = DataLoader(DrugsData(X_test, Y_test), batch_size = batch_size, num_workers = 0, shuffle = False)
    model_dict = torch.load(path)
    model = model_dict['model']
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    ypred,ytrue,scores = [],[],[]
    with torch.no_grad():
        for batch_idx,(X1, X2, target) in enumerate(test_loader):
            input1, input2, target = map(lambda x: Variable(x), [X1, X2, target])
            target = target.long()
            data = torch.stack([input1,input2],1)
            out1, output = model(data)
            pred = torch.argmax(output.data, 1)
            pred = pred.long()
            ypred.extend(pred.tolist())
            scores.extend(output.tolist())
            ytrue.extend(target.tolist())
            test_loss += model.feature_layer.computeLoss(lamda, mean, batch_idx, batch_size) + F.nll_loss(torch.log(output), target)
            pred = output.data.max(1, keepdim = True)[1]  # get the index of the max log-probability
            pred = pred.long()
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        scores = np.asarray(scores)
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
            test_loss/len(test_loader.dataset), correct.item(), len(test_loader.dataset),
            correct.item() / len(test_loader.dataset)))
              
    te_precision, te_recall, te_f1_score,_ = precision_recall_fscore_support(ytrue, ypred, average = 'weighted', 
                                                                             labels = [0,1])
    test_auc = roc_auc_score(ytrue, scores[:,1], average = 'weighted', labels = [0,1])
    scores_dict['test_auc'] = test_auc
    scores_dict['test_f1'] = te_f1_score
    scores_dict['test_pr'] = te_precision
    scores_dict['test_rec'] = te_recall
    scores_dict['test_accuracy'] = accuracy_score(ytrue, ypred)
    pred_df = pd.DataFrame(Y_test.values.tolist(),columns = ['ytrue'])
    pred_df['ypred'] = ypred
    for i in range(scores.shape[1]):
        pred_df['scores' + str(i)] = scores[:,i]
    scores_dict['obt 1\'s'] = pred_df.loc[(pred_df['ypred']==1) & (pred_df['ypred'] == pred_df['ytrue'])].shape[0]
    scores_dict['total 1\'s'] = pred_df.loc[pred_df['ypred']==1].shape[0]
    scores_df = pd.DataFrame.from_dict(scores_dict, orient = 'index').T.reset_index().drop(columns = ['index'])
    if not os.path.exists(results_file_name):
        scores_df.to_csv(results_file_name, sep = ',', index = None)
    else:
        scores_df.to_csv(results_file_name, mode = 'a', sep = ',', index = None)
    print('Test Accuracy : ', accuracy_score(ytrue, ypred))
    print('Test F1_Score : ', te_f1_score)
    print('Test AUC : ', test_auc)
    pred_df.to_csv(pred_file_name, index = None)


# In[7]:


def createParamPath(epochs, batch_size, lamda, mean, lr, ks_f, wd, mpl_ks, maxpools, amsFlg, activFunc,
                    loss_func, atom_ratio, xstep_flg, zstep_flg, ntree, tree_depth, tree_feat_rate, of_fc):
    out_pl1, ks1, out_pl2, ks2 = ks_f
    maxpl1, maxpl2 = maxpools
    if custom_batch_size_flag == True:
        param_path = 'ep_' + str(epochs) + '_bs' + str(batch_size) + '_lam' + str(lamda) + '_mu' + str(mean) +                 '_lr' + str(lr) + '_wd' + str(wd) +  '_ks1' + str(ks1) + '_ks2' + str(ks2) + '_opl1' +                 str(out_pl1) + '_opl2' + str(out_pl2) + '_mpl1' + str(maxpl1)[0] + '_mpl2' + str(maxpl2)[0] +             '_ams' + str(amsFlg)[0] + '_atmrat' + str(atom_ratio) + '_af' + activFunc + '_loss' + loss_func + '_x' +         str(xstep_flg)[0] + '_z' + str(zstep_flg)[0] + '_ntr' + str(ntree) + '_td' + str(tree_depth) + '_tfr' + str(tree_feat_rate) + '_of_fc' + str(of_fc) 
    else:
        param_path = 'ep_' + str(epochs)  + '_lam' + str(lamda) + '_mu' + str(mean) +                 '_lr' + str(lr) + '_wd' + str(wd) +  '_ks1' + str(ks1) + '_ks2' + str(ks2) + '_opl1' +                 str(out_pl1) + '_opl2' + str(out_pl2) + '_mpl1' + str(maxpl1)[0] + '_mpl2' + str(maxpl2)[0] +             '_ams' + str(amsFlg)[0] + '_atmrat' + str(atom_ratio) + '_af' + activFunc + '_loss' + loss_func + '_x' +         str(xstep_flg)[0] + '_z' + str(zstep_flg)[0] + '_ntr' + str(ntree) + '_td' + str(tree_depth) + '_tfr' + str(tree_feat_rate) + '_of_fc' + str(of_fc) 
    import regex as re
    param_path = re.sub(r'[^\P{P}_]+',"",param_path)
    return param_path


# In[9]:


activation_funcs = {
    'relu' : nn.ReLU(inplace = True),
    'selu' : nn.SELU(inplace = True),
    'leaky_relu' : nn.LeakyReLU(inplace = True),
    'tanh' : nn.Tanh(),
    'softmax' : nn.Softmax(dim = 1),
    'sigmoid' : nn.Sigmoid()
}

loss_funcs = {
    'cross_entropy' : nn.CrossEntropyLoss(),
    'hinge' : nn.HingeEmbeddingLoss(),
    'nll' : nn.NLLLoss()
}


# In[8]:


str_fl_nm = '_a1a2b1'
cfg.data_df = getData('../data/smiles_sign_vec_data' + str_fl_nm + '.csv')
print('cfg.data_df : ', cfg.data_df.head())
drug_ids_list = cfg.data_df['DrugBank ID'].unique().tolist()
cfg.feats_list = cfg.data_df.columns.tolist()
cfg.feats_list.remove('DrugBank ID')
cfg.data_df = cfg.data_df[cfg.feats_list + ['DrugBank ID']]



# In[11]:


data_path = "../data/"
base_path = '../' 
loss_func = 'nll'
model_base_path = base_path + 'models/Dcf_Dtf_l2_smi_sign' + str_fl_nm + '/'
res_base_path = base_path + 'Results/' 
base_string = 'dtfl2' + str_fl_nm 


dcf_config = dcf_model_configs()
custom_batch_size_flag = True
config_result_dict = {}
save_flag = 0
seed = 42
mode = 'train'
# mode = 'classify'
for idx, config in enumerate(dcf_config):
    t0 = time.time()
    log_interval = 1 
    cnt = 0 
    epochs, batch_size, lamda_mean, lr, ks_f, wd, mpl_ks, maxpools, activFunc, atom_ratio, amsFlg, xstep_flg, zstep_flg, of_fc, ntree, tree_depth, tree_feat_rate = config
    out_pl1, ks1, out_pl2, ks2 = ks_f
    maxpl1, maxpl2 = maxpools
    lamda, mean = lamda_mean
    forest_config = [ntree, tree_depth, tree_feat_rate]
    config = config[:-3]
    print(config)
    param_path = createParamPath(epochs, batch_size, lamda, mean, lr, ks_f, wd, mpl_ks, maxpools, amsFlg, 
                                 activFunc, loss_func, atom_ratio, xstep_flg, zstep_flg, ntree, tree_depth, tree_feat_rate, of_fc)
    print(param_path)
    X_train, X_test, Y_train, Y_test = getTrainTestDataFromFile()
    print(Y_train.loc[Y_train['Interaction']==1].shape)
    print(Y_train.groupby('Interaction').size())
    print(Y_test.groupby('Interaction').size())
    print('X_train.shape : {}, Y_train.shape : {}'.format(X_train.shape, Y_train.shape))
    print('X_test.shape : {}, Y_test.shape : {}'.format(X_test.shape, Y_test.shape))
    model_path = model_base_path + base_string + '_model' + param_path + '_reg' + str(idx) + '.pth'
    
    test_results_dict = {}
    if not custom_batch_size_flag:
        batch_size = X_train.shape[0]
    
    t00 = time.time()
    
    ##Training and testing 
    if mode == 'train':
        scores_dict = train_on_batch(X_train, Y_train, config, forest_config)
        t01 = time.time()
        print('Time taken for Training is : {}'.format(str(dtym.timedelta(seconds = t01 - t00))))
        # Store data (serialize)
        with open(res_base_path + param_path + '.pickle', 'wb') as handle:
            pickle.dump(scores_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
        test(seed,X_test,Y_test,model_path,scores_dict,idx+1)
        t02 = time.time()
        print('Time taken for Testing is : {}'.format(str(dtym.timedelta(seconds = t02 - t01))))
    elif mode == 'classify':
        with open(res_base_path + param_path + '.pickle', 'rb') as handle:
              scores_dict = pickle.load(handle)
        t00 = time.time()
        test(seed,X_test,Y_test,model_path,scores_dict,idx+1)
        t01 = time.time()
        print('Time taken for Testing is : {}'.format(str(dtym.timedelta(seconds = t01 - t00))))





