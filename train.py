#import imp
from mini_batch_loader import *
import MyFCN_de
import sys
import time
import State_de
import pixelwise_a3c_de
import os
import torch
import Myloss
import pixelwise_a3c_el
import MyFCN_el
from models import FFDNet
import torch.nn as nn
from mini_batch_loader import MiniBatchLoader 
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import copy

from model.model import *
#_/_/_/ paths _/_/_/ 

TRAINING_DATA_PATH          = "./data/label.txt"
TESTING_DATA_PATH           = "./data/label.txt"
IMAGE_DIR_PATH              = "./our485/low/"
SAVE_PATH            = "./model/fpop_myfcn_"
TRAINING_DATA_PATH          = "./train/label.txt"
TESTING_DATA_PATH           = "./train2/label.txt"
IMAGE_DIR_PATH              = "./train2/"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.0002
TRAIN_BATCH_SIZE = 2
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES = 30000
EPISODE_LEN = 10
SNAPSHOT_EPISODES  = 100
TEST_EPISODES = 50000
GAMMA = 1.05 # discount factor

#noise setting


N_ACTIONS = 27
MOVE_RANGE = 27 #number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 244
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
GPU_ID = 0
 
 
def main(fout):
    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH, 
        TRAINING_DATA_PATH,
        IMAGE_DIR_PATH, 
        CROP_SIZE)
    pixelwise_a3c_el.chainer.cuda.get_device_from_id(GPU_ID).use()
   
    
    # load ffdnet
    in_ch = 3
    model_fn = 'FFDNet_models/net_rgb.pth'
    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                            model_fn)
    # Create model
    print('Loading model ...\n')
    net = FFDNet(num_input_channels=in_ch)

    # Load saved weights

    state_dict = torch.load(model_fn)
    device_ids = [GPU_ID]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    model.load_state_dict(state_dict)
    model.eval()
    current_state = State_de.State_de((TRAIN_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE, model)

    # load myfcn model
    model_el = MyFCN_el.MyFcn(N_ACTIONS)

    # _/_/_/ setup _/_/_/
    optimizer_el = pixelwise_a3c_el.chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer_el.setup(model_el)

    agent_el = pixelwise_a3c_el.PixelWiseA3C(model_el, optimizer_el, EPISODE_LEN, GAMMA)
    agent_el.model.to_gpu()

    # load myfcn model for de
    model_de = MyFCN_de.MyFcn_denoise(2)

    # _/_/_/ setup _/_/_/

    optimizer_de = pixelwise_a3c_de.chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer_de.setup(model_de)

    agent_de = pixelwise_a3c_de.PixelWiseA3C(model_de, optimizer_de, EPISODE_LEN, GAMMA)
    agent_de.model.to_gpu()

    # NIMA model
    base_model = models.vgg16(pretrained=True)
    NIMA_model = NIMA(base_model)
    NIMA_model.load_state_dict(torch.load("premodel/epoch-82.pth"))
    seed = 42
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NIMA_model = NIMA_model.to(device)

    NIMA_model.eval()
    test_transform = transforms.Compose([
    transforms.RandomCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])


    #_/_/_/ training _/_/_/
 
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    L_color = Myloss.L_color()
    L_color_rate = Myloss.L_color_rate()
    L_spa = Myloss.L_spa()
    L_TV = Myloss.L_TV()
    L_exp = Myloss.L_exp(16, 0.6)
    for episode in range(1, N_EPISODES+1):
        # display current episode
        print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()
        # load images
        r = indices[i:i+TRAIN_BATCH_SIZE]
        #print(r)
        raw_x = mini_batch_loader.load_training_data(r)
        current_state.reset(raw_x)
        reward_de = np.zeros(raw_x.shape, raw_x.dtype)
        action_value = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0
        premean = 0.0
        for t in range(0, EPISODE_LEN):
            raw_tensor = torch.from_numpy(raw_x).cuda()
            previous_image = current_state.image.copy()
            action_el = agent_el.act_and_train(current_state.image, reward_de)
            #print(action_el)
            action_value = (action_el - 9)/18
            current_state.step_el(action_el)
            action_de = agent_de.act_and_train(current_state.image, reward_de)
            current_state.step_de(action_de)
            currentImg1 = (current_state.image[0, ::])
            currentImg2 = (current_state.image[1, ::])
            currentImg1 = np.transpose(currentImg1, (1, 2, 0))
            currentImg2 = np.transpose(currentImg2, (1, 2, 0))
            currentImg1 -= [0.485, 0.456, 0.406]
            currentImg1 /= [0.229, 0.224, 0.225]
            currentImg2 -= [0.485, 0.456, 0.406]
            currentImg2 /= [0.229, 0.224, 0.225]
            currentImg1 = np.transpose(currentImg1, (2, 0, 1))
            currentImg2 = np.transpose(currentImg2, (2, 0, 1))
            currentImg1 = torch.tensor(currentImg1)
            currentImg2 = torch.tensor(currentImg2)
            currentImg1 = currentImg1.unsqueeze(dim=0)
            currentImg2 = currentImg2.unsqueeze(dim=0)
            imt = torch.cat((currentImg1, currentImg2), 0)
            imt = imt.to(device)
            mean = 0.0
            with torch.no_grad():
                out = NIMA_model(imt)
            out = out.view(20, 1)
            for j, e in enumerate(out, 1):
                if j % 10 == 0:
                    mean += 10 * e
                else:
                    mean += (j % 10) * e
            
            
            previous_image_tensor = torch.from_numpy(previous_image).cuda()
            current_state_tensor = torch.from_numpy(current_state.image).cuda()
            action_tensor = torch.from_numpy(action_value).cuda()
            loss_spa_cur = torch.mean(L_spa(current_state_tensor, raw_tensor))
            loss_col_cur = 50 * torch.mean(L_color(current_state_tensor))
            Loss_TV_cur = 200 * L_TV(action_tensor)
            loss_exp_cur = 80 * torch.mean(L_exp(current_state_tensor))
            loss_col_rate_pre = 20 * torch.mean(L_color_rate(previous_image_tensor, current_state_tensor))
            reward_current = loss_col_cur + loss_spa_cur + loss_exp_cur + Loss_TV_cur - 2 * (mean - premean)
            premean = mean
            reward = - reward_current
            reward_de = reward.cpu().numpy()
            sum_reward += np.mean(reward_de) * np.power(GAMMA, t)

        agent_el.stop_episode_and_train(current_state.image, reward_de, True)
        agent_de.stop_episode_and_train(current_state.image, reward_de, True)

        print("train total reward {a}".format(a=sum_reward))
        fout.write("train total reward {a}\n".format(a=sum_reward))
        sys.stdout.flush()

        if episode % SNAPSHOT_EPISODES == 0:
            agent_el.save(SAVE_PATH+str(episode))
        
        if i+TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:        
            i += TRAIN_BATCH_SIZE

        if i+2*TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        # optimizer_de.alpha = LEARNING_RATE*((1-episode/N_EPISODES)**0.9)
 
     
 
if __name__ == '__main__':
    #try:
    fout = open('log_ex7.txt', "w")
    start = time.time()
    main(fout)
    end = time.time()
    print("{s}[s]".format(s=end - start))
    print("{s}[m]".format(s=(end - start)/60))
    print("{s}[h]".format(s=(end - start)/60/60))
    fout.write("{s}[s]\n".format(s=end - start))
    fout.write("{s}[m]\n".format(s=(end - start)/60))
    fout.write("{s}[h]\n".format(s=(end - start)/60/60))
    fout.close()
    #except Exception as error:
        #print("error!")
