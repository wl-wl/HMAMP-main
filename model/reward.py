# -*- coding:utf-8 -*-
import os
import random
import math
import copy
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import re

class PolicyGradient(object):
    """ policy gradient"""
    def __init__(self,gen_loader,beta=0):
        self.data_loader = gen_loader
        self.beta = beta

    def get_reward(self,x,discriminator1,discriminator2,use_cuda):
        """
        Args:
            x : a generated sequence
            discriminator : discrimanator model
            use_cuda: return torch.tensor or torch.tensor.cuda
            gamma: decay coefficiency
        """
        seq_len = len(x)
        if seq_len == 2 :
            rewards = np.array([-1])
        else :
            rewards = [0]*(seq_len-1)
            if x[1:].find('<') != -1:
                pos = []
                for idx,char in enumerate(x):
                    if idx != 0 and char == '<':
                        pos.append(idx)
                for idx in pos:
                    rewards[idx-1]=-1
            else :
                x_temp=x
                nadir_slack=0.05
                mic_predict=discriminator1.classify(self.data_loader.char_tensor(x_temp))
                hemo_predict=discriminator2.classify(self.data_loader.char_tensor(x_temp))
                # reward = discriminator1.classify(self.data_loader.char_tensor(x_temp)) + discriminator2.classify(self.data_loader.char_tensor(x_temp)) - self.beta * abs(discriminator1.classify(self.data_loader.char_tensor(x_temp)) - discriminator2.classify(self.data_loader.char_tensor(x_temp)))
                reward = min(discriminator1.classify(self.data_loader.char_tensor(x_temp)) , discriminator2.classify(
                    self.data_loader.char_tensor(x_temp)))
                nadir=reward*nadir_slack
                reward=reward-nadir
                # reward = 0.5*math.log(discriminator1.classify(self.data_loader.char_tensor(x_temp)))+0.5*math.log(discriminator2.classify(
                #     self.data_loader.char_tensor(x_temp)))
                for i in range(len(rewards)):
                    rewards[i] = reward
        if use_cuda :
            return torch.Tensor(rewards).cuda(),mic_predict,hemo_predict
        else :
            return torch.Tensor(rewards),mic_predict,hemo_predict
