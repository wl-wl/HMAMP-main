# -*- coding:utf-8 -*-
import os
import sys
import random
import math
import tqdm
import argparse
import joblib
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



from dataset import GeneratorData, DiscriminatorData
from model.generator import Generator
from model.discriminator import Discriminator
from model.reward import PolicyGradient  # Rollout
from utils import read_peptides_from_file, GANLoss, get_reward


# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=0, type=int)
parser.add_argument('--use_cuda', action='store', default=None, type=int)

parser.add_argument('--g_layers', action='store', default=2, type=int)
parser.add_argument('--g_embed_size', action='store', default=300, type=int)
parser.add_argument('--g_hidden_size', action='store', default=1024, type=int)
parser.add_argument('--g_lr', action='store', default=0.000001, type=float)
parser.add_argument('--g_temperature', action='store', default=1, type=float)
parser.add_argument('--g_temperature_decay', action='store', default=0.995, type=float)

parser.add_argument('--d_embed_size', action='store', default=100, type=int)
parser.add_argument('--d_hidden_size', action='store', default=300, type=int)
parser.add_argument('--d_lr', action='store', default=0.00001, type=float)
parser.add_argument('--d_dropout', action='store', default=0.2, type=float)
parser.add_argument('--d_batch_size', action='store', default=64, type=int)
parser.add_argument('--d_pre_epoch', action='store', default=15, type=int)

parser.add_argument('--epochs_num', action='store', default=50, type=int)  # 50
parser.add_argument('--g_epoch', action='store', default=1000, type=int) #1000
parser.add_argument('--d_epoch', action='store', default=20, type=int)

parser.add_argument('--print_every', action='store', default=100, type=int)
parser.add_argument('--beta', action='store', default=0.3, type=int)

parser.add_argument('--model_path', action='store', default='model_saved/PTG', type=str)
parser.add_argument('--save_path', action='store', default='model_saved/PolicyGradient/RES',
                    type=str)
parser.add_argument('--g_name', action='store', default='MLE_generator.pt', type=str)
parser.add_argument('--g_data_file', action='store', default='amp2.txt', type=str)
parser.add_argument('--train_file_1', action='store', default='amp_b_2(mic).txt', type=str)
parser.add_argument('--train_file_2', action='store', default='hemo2.txt', type=str)

parser.add_argument('--valid_file_1', action='store', default='amp_b_2(mic).txt', type=str)
parser.add_argument('--valid_file_2', action='store', default='hemo2.txt', type=str)



opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if opt.use_cuda is None:
    opt.use_cuda = torch.cuda.is_available()

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)

word_index_dict = {'b': '0', 'a': '1', 'r': '2', 'n': '3', 'd': '4', 'c': '5', 'q': '6', 'e': '7', 'g': '8', 'h': '9',
                   'i': '10', 'l': '11', 'k': '12', 'm': '13', 'f': '14', 'p': '15', 's': '16', 't': '17', 'w': '18',
                   'y': '19', 'v': '20', 'x': '21', 'z': '22', 'u': '23', 'j': '24', 'o': '25'}
index_word_dict = {'0': 'b', '1': 'a', '2': 'r', '3': 'n', '4': 'd', '5': 'c', '6': 'q', '7': 'e', '8': 'g', '9': 'h',
                   '10': 'i', '11': 'l', '12': 'k', '13': 'm', '14': 'f', '15': 'p', '16': 's', '17': 't', '18': 'w',
                   '19': 'y', '20': 'v', '21': 'x', '22': 'z', '23': 'u', '24': 'j', '25': 'o'}



# char_num = len(tokens)
char_num = len(word_index_dict)

optimizer_instance = torch.optim.Adam

if __name__ == '__main__':
    # Define Networks
    print('---------- Defining and loading Network')
    generator = Generator(input_size=char_num, embed_size=opt.g_embed_size, hidden_size=opt.g_hidden_size,
                          output_size=char_num, n_layers=opt.g_layers, use_cuda=opt.use_cuda,
                          optimizer_instance=optimizer_instance, lr=opt.g_lr)
    discriminator_1 = Discriminator(input_size=char_num, embed_size=opt.d_embed_size, hidden_size=opt.d_hidden_size,
                                    use_cuda=opt.use_cuda, dropout=opt.d_dropout, lr=opt.d_lr,
                                    optimizer_instance=torch.optim.Adam)
    discriminator_2 = Discriminator(input_size=char_num, embed_size=opt.d_embed_size, hidden_size=opt.d_hidden_size,
                                    use_cuda=opt.use_cuda, dropout=opt.d_dropout, lr=opt.d_lr,
                                    optimizer_instance=torch.optim.Adam)
    # fcd_model = load_ref_model()
    # Load pretrained generator
    generator.load_model(os.path.join(os.getcwd(), opt.model_path, opt.g_name), map_location='cuda:0')

    # Define path
    if not os.path.exists(os.path.join(os.getcwd(), opt.save_path)):
        os.makedirs(os.path.join(os.getcwd(), opt.save_path))
    reward_d1_path = os.path.join(os.getcwd(), opt.save_path, 'A_reward_d1.txt')
    reward_d2_path = os.path.join(os.getcwd(), opt.save_path, 'A_reward_d2.txt')
    reward_all_path = os.path.join(os.getcwd(), opt.save_path, 'A_reward_all.txt')
    reward_all_path2 = os.path.join(os.getcwd(), opt.save_path, 'A_reward_all2.txt')
    reward_valid_path = os.path.join(os.getcwd(), opt.save_path, 'A_reward_valid.txt')
    nll_1_path = os.path.join(os.getcwd(), opt.save_path, 'A_nll_1.txt')
    nll_2_path = os.path.join(os.getcwd(), opt.save_path, 'A_nll_2.txt')

    model_d1_path = os.path.join(os.getcwd(), opt.save_path, 'A_discriminator_1.pt')
    model_d2_path = os.path.join(os.getcwd(), opt.save_path, 'A_discriminator_2.pt')
    model_g_path = os.path.join(os.getcwd(), opt.save_path, 'A_generator.pt')
    log_path_d1 = os.path.join(os.getcwd(), opt.save_path, 'A_dis_1_loss.txt')
    log_path_d2 = os.path.join(os.getcwd(), opt.save_path, 'A_dis_2_loss.txt')
    log_path_d3 = os.path.join(os.getcwd(), opt.save_path, 'A_dis_all_loss.txt')

    gloss_path=os.path.join(os.getcwd(), opt.save_path, 'A_gen_all_loss.txt')
    sample_path = os.path.join(os.getcwd(), opt.save_path, 'sample.txt')
    predict_all_path= os.path.join(os.getcwd(), opt.save_path, 'predict_all.txt')

    # Define GeneratorData
    print('---------- Loading GeneratorData')
    gen_loader = GeneratorData(os.path.join(os.getcwd(), 'data', opt.g_data_file), tokens=tokens, use_cuda=opt.use_cuda)
    eval_loader_1 = GeneratorData(os.path.join(os.getcwd(), 'data', opt.valid_file_1), tokens=tokens,
                                  use_cuda=opt.use_cuda)
    eval_loader_2 = GeneratorData(os.path.join(os.getcwd(), 'data', opt.valid_file_2), tokens=tokens,
                                  use_cuda=opt.use_cuda)

    # Read truth data for discriminator
    truth_data_path_1 = os.path.join(os.getcwd(), 'data', opt.train_file_1)
    truth_data_path_2 = os.path.join(os.getcwd(), 'data', opt.train_file_2)
    truth_data_1, _ = read_peptides_from_file(truth_data_path_1)
    truth_data_2, _ = read_peptides_from_file(truth_data_path_2)

    # Use Generator to generate some fake data for discriminator pretraining
    fake_data = []
    num = 0
    fake_data_len = 10000  # 10000
    print('---------- Using generator to generate fake data for discriminiator pretraining')
    with torch.no_grad():
        while (num < fake_data_len):
            sample = generator.generate(gen_loader)
            if len(sample)>10:
                fake_data.append(sample)
                num = num + 1

    # evaluate pretrained generator
    print('---------- Evaluating pretrained generator')


    real_sample_1 = eval_loader_1.peptide_list
    real_sample_2 = eval_loader_2.peptide_list


    # Define DiscriminatorData
    print('---------- Loading DiscriminatorData')
    random.shuffle(fake_data)
    dis_loader1 = DiscriminatorData(truth_data=truth_data_1, fake_data=fake_data[0:len(truth_data_1)], tokens=tokens,
                                    batch_size=opt.d_batch_size)
    random.shuffle(fake_data)
    dis_loader2 = DiscriminatorData(truth_data=truth_data_2, fake_data=fake_data[0:len(truth_data_2)], tokens=tokens,
                                    batch_size=opt.d_batch_size)
    # Pretrain Discriminator
    print('---------- Pretrain Discriminator ...')
    discriminator_1.train()
    discriminator_2.train()
    loss_1 = discriminator_1.train_epochs(dis_loader1, opt.d_pre_epoch)
    loss_2 = discriminator_2.train_epochs(dis_loader2, opt.d_pre_epoch)
    f = open(log_path_d1, 'a')
    for l in loss_1:
        f.write(str(l) + "\n")
    f.close()
    f = open(log_path_d2, 'a')
    for l in loss_2:
        f.write(str(l) + "\n")
    f.close()

    # Adversarial Training
    policy = PolicyGradient(gen_loader, beta=opt.beta)
    if opt.use_cuda:
        ganloss = GANLoss().cuda()
    else:
        ganloss = GANLoss()
    print('---------- Start Adeversatial Training...')
    sample_list = []
    sample_mic_list=[]
    sample_hemo_list=[]
    HV_list=[]
    HV_b_list=[]
    HV_w_list=[]
    g_loss_list=[]
    g_loss_list1=[]
    for epoch in range(opt.epochs_num):
        # Train the generator for g
        discriminator_1.eval()
        discriminator_2.eval()
        print('---------- Training generator')
        total_d1 = 0
        total_d2 = 0
        total_valid = 0
        vb, vw=0,0
        sample_count=0
        sample_mic_count=0
        sample_hemo_count=0
        generator.optimizer.zero_grad()
        for i in range(opt.g_epoch):
            # generate a sample
            with torch.no_grad():
                sample = generator.generate(gen_loader)

                # calculate the reward
                # print(sample)
                reward,mic_predict,hemo_predict = policy.get_reward(x=sample, discriminator1=discriminator_1, discriminator2=discriminator_2,
                                           use_cuda=opt.use_cuda)
                reward_d1, reward_d2, reward_valid = get_reward(sample, discriminator_1, discriminator_2, gen_loader)
                total_d1 = total_d1 + reward_d1
                total_d2 = total_d2 + reward_d2
                total_valid = total_valid + reward_valid
                f = open(reward_all_path2, 'a')
                f.write(str(reward_d1) + ',' + str(reward_d2) + "\n")
                f.close()
                if epoch == opt.epochs_num - 1:
                    f = open(predict_all_path, 'a')
                    f.write(str(mic_predict) + ',' + str(hemo_predict) + "\n")
                    f.close()

            # calculate the loss and optimize
            prob = generator.get_prob(gen_loader.char_tensor(sample))
            g_loss = ganloss(prob, reward)
            # g_loss+=hemo_loss
            g_loss_list.append(g_loss)
            g_loss+=(vb-vw)**2
            g_loss_list1.append(g_loss)
            f = open(gloss_path, 'a')
            f.write(str(g_loss) + "\n")
            f.close()

            g_loss.backward()
            generator.optimizer.step()
            generator.optimizer.zero_grad()
            if (i + 1) % opt.print_every == 0 and i != 0:
                print('Adversatial epoch:{}/{}, Generator epoch:{}/{}'.format(epoch + 1, opt.epochs_num, i + 1,
                                                                              opt.g_epoch))
                print('Reward:{}  {}  {}'.format(total_d1 / opt.print_every, total_d2 / opt.print_every,
                                                 total_valid / opt.print_every))
                # save reward to log file
                f = open(reward_d1_path, 'a')
                f.write(str(total_d1 / opt.print_every) + "\n")
                f.close()
                f = open(reward_d2_path, 'a')
                f.write(str(total_d2 / opt.print_every) + "\n")
                f.close()
                f = open(reward_valid_path, 'a')
                f.write(str(total_valid / opt.print_every) + "\n")
                f.close()
                f = open(reward_all_path, 'a')
                f.write(str(total_d1)+','+str(total_d2)+ "\n")
                f.close()

                total_d1 = 0
                total_d2 = 0
                total_valid = 0



        # evaluate generator for nll
        print('---------- Evaluating generator with nll ...')
        with torch.no_grad():
            mean_nll_1 = generator.evaluate(eval_loader=eval_loader_1, log_path=nll_1_path)
            mean_nll_2 = generator.evaluate(eval_loader=eval_loader_2, log_path=nll_2_path)
        print('Mean negative log likelihood on eval_dataset_1: {}'.format(mean_nll_1))
        print('Mean negative log likelihood on eval_dataset_2: {}'.format(mean_nll_2))

        # Train the discriminator
        # generate fake data for discriminator
        print('---------- Generating fake data for discriminator ...')
        fake_data = []
        num = 0
        with torch.no_grad():
            while (num < fake_data_len):
                sample = generator.generate(gen_loader)
                if len(sample)>10:
                    fake_data.append(sample)
                    num = num + 1




        print('---------- Train Discriminator ...')
        truth_data_path_1 = os.path.join(os.getcwd(), 'data', opt.train_file_1)
        truth_data_path_2 = os.path.join(os.getcwd(), 'data', opt.train_file_2)
        truth_data_1, _ = read_peptides_from_file(truth_data_path_1)
        truth_data_2, _ = read_peptides_from_file(truth_data_path_2)

        # random.shuffle(fake_data)
        # dis_loader1.update(truth_data=None, fake_data=fake_data[0:len(truth_data_1)])
        # random.shuffle(fake_data)
        # dis_loader2.update(truth_data=None, fake_data=fake_data[0:len(truth_data_2)])
        random.shuffle(fake_data)
        dis_loader1 = DiscriminatorData(truth_data=truth_data_1, fake_data=fake_data[0:len(truth_data_1)],
                                        tokens=tokens,
                                        batch_size=opt.d_batch_size)
        random.shuffle(fake_data)
        dis_loader2 = DiscriminatorData(truth_data=truth_data_2, fake_data=fake_data[0:len(truth_data_2)],
                                        tokens=tokens,
                                        batch_size=opt.d_batch_size)

        discriminator_1.train()
        discriminator_2.train()
        loss_1 = discriminator_1.train_epochs(dis_loader1, opt.d_epoch)
        loss_2 = discriminator_2.train_epochs(dis_loader2, opt.d_epoch)
        f = open(log_path_d1, 'a')
        for l in loss_1:
            f.write(str(l) + "\n")
        f.close()
        f = open(log_path_d2, 'a')
        for l in loss_2:
            f.write(str(l) + "\n")
        f.close()
        f = open(log_path_d3, 'a')
        for i in range(len(loss_2)):
            f.write(str(loss_1[i]) + ','+str(loss_2[i])+"\n")
        f.close()
        discriminator_1.save_model(model_d1_path)
        discriminator_2.save_model(model_d2_path)
print('sample_list',sample_list)
print('sample_hemo_list',sample_hemo_list)
print('sample_mic_list',sample_mic_list)
print('HV_list',HV_list)
print('HV_w_list',HV_w_list)
print('HV_b_list',HV_b_list)
