# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dataset_NormalizeSpace import Dataset
from models.funsr import funsr
from models.discriminator import Discriminator
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh
from models.utils import get_root_logger, print_log
import math
import mcubes
import warnings
import matplotlib.pyplot as plt
import re
# from visdom import Visdom
warnings.filterwarnings("ignore")
import torch.nn as nn
from datetime import datetime

################### FUNSR Implementation ##################################3
class Runner:
    def __init__(self, args, conf_path, mode='train'):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.np_data_name'] = self.conf['dataset.np_data_name']
        self.base_exp_dir = self.conf['general.base_exp_dir'] + args.dir
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        
        self.dataset_np = Dataset(self.conf['dataset'], args.dataname)
        self.dataname = args.dataname
        self.iter_step = 0
        # momentum 
        self.betas = (0.9, 0.999)
        # Training parameters
        self.maxiter = self.conf.get_int('train.maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.eval_num_points = self.conf.get_int('train.eval_num_points')
        self.labmda_scc = self.conf.get_float('train.labmda_scc')
        self.labmda_adl = self.conf.get_float('train.labmda_adl')

        self.mode = mode

        # Networks
        self.sdf_network = funsr(**self.conf['model.sdf_network']).to(self.device)
        self.discriminator = Discriminator(**self.conf['model.discriminator']).to(self.device)
        self.sdf_optimizer = torch.optim.Adam(self.sdf_network.parameters(), lr=self.learning_rate)
        self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate,betas=self.betas)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()
       

    def train(self):
        timestamp_start = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        print(f"Start time:{timestamp_start}")
        log_file = os.path.join(os.path.join(self.base_exp_dir), 'logger.log')

        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        batch_size = self.batch_size

        res_step = self.maxiter - self.iter_step
        ########Live losss visualization ########################
        # viz = Visdom()
        # viz.line([0.,],[0.,],win = 'train_loss', opts=dict(title = 'train loss'))
        # count = 0

        for iter_i in tqdm(range(res_step)):
                self.update_learning_rate_np(iter_i)

                points, samples, point_gt = self.dataset_np.np_train_data(batch_size)

                ###########Train FUNSR Network################
                self.sdf_optimizer.zero_grad()
                samples.requires_grad = True
                gradients_sample = self.sdf_network.gradient(samples).squeeze() # 5000x3
                sdf_sample = self.sdf_network.sdf(samples)                      # 5000x1
                grad_norm = F.normalize(gradients_sample, dim=1)                
                sample_moved = samples - grad_norm * sdf_sample                 

                ######SDF Loss#########################
                loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1).mean()


                ############Sign consistency constrain Loss################# 
                SCC = F.normalize(sample_moved-points, dim=1)                
                loss_SCC = (1.0 - F.cosine_similarity(grad_norm, SCC, dim=1)).mean()

                loss = loss_sdf
                
                ##################funsr Loss ####################
                G_loss = loss  + loss_SCC *self.labmda_scc
                
                #############Train Discriminator #################
                self.dis_optimizer.zero_grad()
                d_fake_output = self.discriminator.sdf(sdf_sample.detach())
                d_fake_loss=self.get_discriminator_loss_single(d_fake_output,label=False)
                
                real_sdf = torch.zeros(points.size(0), 1).to(self.device)
                d_real_output = self.discriminator.sdf(real_sdf)
                d_real_loss=self.get_discriminator_loss_single(d_real_output,label=True)
                dis_loss = d_real_loss + d_fake_loss
                dis_loss.backward()
                self.dis_optimizer.step()


                ################Total Loss ################
                d_fake_output = self.discriminator.sdf(sdf_sample)
                gan_loss=self.get_funsr_loss(d_fake_output)
                total_loss = gan_loss* self.labmda_adl + G_loss
                total_loss.backward()
                self.sdf_optimizer.step()
                
                #############Loss visualization using Visdom #################
                # count = count +1
                # viz.line([total_loss.item()],[count],win = 'train_loss',update = 'append')    

                ############# Saving #################
                self.iter_step += 1
                if self.iter_step % self.report_freq == 0:
                    print_log('iter: {:8>d} total_loss = {} lr = {}'.format(self.iter_step, total_loss, self.sdf_optimizer.param_groups[0]['lr']), logger=logger)


                if self.iter_step % self.val_freq == 0 and self.iter_step!=0: 
                    self.validate_mesh2(resolution=256, threshold=args.mcubes_threshold, point_gt=point_gt, iter_step=self.iter_step, logger=logger)

                if self.iter_step % self.save_freq == 0 and self.iter_step!=0: 
                    self.save_checkpoint()
        
        timestamp_end = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        start_datetime = datetime.strptime(timestamp_start, '%Y%m%d_%H%M%S')
        end_datetime = datetime.strptime(timestamp_end, '%Y%m%d_%H%M%S')
        duration = end_datetime - start_datetime
        duration_in_minutes = duration.total_seconds() / 60 # min


    def validate_mesh2(self, resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None):


        os.makedirs(os.path.join(self.base_exp_dir, 'outputs'), exist_ok=True)
        mesh = self.extract_geometry2(resolution=resolution, threshold=threshold, query_func=lambda pts: -self.sdf_network.sdf(pts))

        mesh.export(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}.ply'.format(self.iter_step,str(threshold))))

    # create cube from (-1,-1,-1) to (1,1,1) and uniformly sample points for marching cube
    def create_cube(self,N):

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)

        # the voxel_origin is the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)
        
        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long().float() / N) % N
        samples[:, 0] = ((overall_index.long().float() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        samples.requires_grad = False

        return samples
        
    #############ADL Loss ##############################
    def get_funsr_loss(self,pred_fake):
        fake_loss=torch.mean((pred_fake-1)**2)
        return fake_loss
    def get_discriminator_loss_single(self,pred,label=True):
        if label==True:
            loss=torch.mean((pred-1)**2)
            return loss
        else:
            loss=torch.mean((pred)**2)
            return loss
    
    def update_learning_rate_np(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        for g in self.sdf_optimizer.param_groups:
            g['lr'] = lr
    
    

    def extract_fields2(self, resolution, query_func):
        N = resolution
        max_batch = 1000000
        # the voxel_origin is the (bottom, left, down) corner, not the middle
        cube = self.create_cube(resolution).cuda()
        cube_points = cube.shape[0]


        with torch.no_grad():
            head = 0
            while head < cube_points:
                
                query = cube[head : min(head + max_batch, cube_points), 0:3]
                
                # inference defined in forward function per pytorch lightning convention
                pred_sdf = query_func(query)

                cube[head : min(head + max_batch, cube_points), 3] = pred_sdf.squeeze()
                    
                head += max_batch

        # for occupancy instead of SDF, subtract 0.5 so the surface boundary becomes 0
        sdf_values = cube[:, 3]
        sdf_values = sdf_values.reshape(N, N, N).detach().cpu()


        return sdf_values


    def extract_geometry2(self, resolution, threshold, query_func):
        print('Creating mesh with threshold: {}'.format(threshold))
        u = self.extract_fields2( resolution, query_func).numpy()
        vertices, triangles = mcubes.marching_cubes(u, threshold)

        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (resolution - 1)

        vertices[:,0] = vertices[:,0]*voxel_size + voxel_origin[0]
        vertices[:,1] = vertices[:,1]*voxel_size + voxel_origin[0]
        vertices[:,2] = vertices[:,2]*voxel_size + voxel_origin[0]

        mesh = trimesh.Trimesh(vertices, triangles)

        return mesh


    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))

        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        
        self.iter_step = checkpoint['iter_step']
            
    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/conf.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcubes_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--dir', type=str, default='T8')
    parser.add_argument('--dataname', type=str, default='T8')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args, args.conf, args.mode)

    if args.mode == 'train':
        runner.train()

