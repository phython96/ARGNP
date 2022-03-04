import os
import sys
import dgl
import torch
import hydra
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from data import *
from models.model_train import *
from utils.utils import *
from utils.visualize import *
from tensorboardX import SummaryWriter
from utils.record_utils import record_run

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import warnings
warnings.filterwarnings('ignore')
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from omegaconf import DictConfig, OmegaConf


class Trainer(object):

    def __init__(self, args):
        
        self.args = args
        self.console = Console()

        self.console.log('=> [0] Initial TensorboardX')
        self.writer = SummaryWriter(comment = f'Task: {args.ds.task}, Data: {args.ds.data}, Geno: {args.ds.load_genotypes}')

        self.console.log('=> [1] Initial Settings')
        np.random.seed(args.basic.seed)
        torch.manual_seed(args.basic.seed)
        torch.cuda.manual_seed(args.basic.seed)
        cudnn.enabled = True

        load_genotypes = os.path.join(get_original_cwd(), args.ds.load_genotypes)
        self.console.log('=> [2] Initial Models')
        if not os.path.isfile(load_genotypes):
            raise Exception('Genotype file not found!')
        else:
            with open(load_genotypes) as f:
                genotypes      = eval(f.read())
                args.nb_layers = len(genotypes)
                args.nb_nodes  = len({ x for x, a, b, c in genotypes[0].V})
        self.metric    = load_metric(args)
        self.loss_fn   = get_loss_fn(args).cuda()
        trans_input_fn = get_trans_input(args)
        self.model     = Model_Train(args, genotypes, trans_input_fn, self.loss_fn).to("cuda")
        self.console.log(f'[red]=> Subnet Parameters: {count_parameters_in_MB(self.model)} MB')

        self.console.log(f'=> [3] Preparing Dataset')
        self.dataset    = load_data(args)
        if args.ds.pos_encode > 0:
            self.console.log(f'[red]==> [3.1] Adding positional encodings')
            self.dataset._add_positional_encodings(args.ds.pos_encode)
        self.train_data = self.dataset.train
        self.val_data   = self.dataset.val
        self.test_data  = self.dataset.test
        self.load_dataloader()

        self.console.log(f'=> [4] Initial Optimizers')
        if args.optimizer.name == 'SGD':
            self.optimizer   = torch.optim.SGD(
                params       = self.model.parameters(),
                lr           = args.optimizer.lr,
                momentum     = args.optimizer.momentum,
                weight_decay = args.optimizer.weight_decay,
            )
            
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
                optimizer  = self.optimizer,
                T_max      = float(args.basic.train_epochs),
                eta_min    = args.optimizer.lr_min
            )

        elif args.optimizer.name == 'ADAM':
            self.optimizer   = torch.optim.Adam(
                params       = self.model.parameters(),
                lr           = args.optimizer.lr,
                weight_decay = args.optimizer.weight_decay,
            )

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = self.optimizer,
                mode      = 'min',
                factor    = 0.5,
                patience  = args.optimizer.patience,
                verbose   = True
            )
        else:
            raise Exception('Unknown optimizer!')


    def load_dataloader(self):
        
        num_train = int(len(self.train_data) * self.args.basic.data_clip)
        indices   = list(range(num_train))

        self.train_queue = torch.utils.data.DataLoader(
            dataset     = self.train_data,
            batch_size  = self.args.ds.batch,
            pin_memory  = True,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers = self.args.basic.nb_workers,
            collate_fn  = self.dataset.collate,
            drop_last   = True,
        )

        if self.val_data is not None:
            num_valid = int(len(self.val_data) * self.args.basic.data_clip)
            indices   = list(range(num_valid))

            self.val_queue  = torch.utils.data.DataLoader(
                dataset     = self.val_data,
                batch_size  = self.args.ds.batch,
                pin_memory  = True,
                sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
                num_workers = self.args.basic.nb_workers,
                collate_fn  = self.dataset.collate,
                shuffle     = False
            )

        num_test  = int(len(self.test_data) * self.args.basic.data_clip)
        indices   = list(range(num_test))

        self.test_queue = torch.utils.data.DataLoader(
            dataset     = self.test_data,
            batch_size  = self.args.ds.batch,
            pin_memory  = True,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers = self.args.basic.nb_workers,
            collate_fn  = self.dataset.collate,
            shuffle     = False,
        )

        self.plot_queue = torch.utils.data.DataLoader(
            dataset     = self.test_data,
            batch_size  = 1,
            pin_memory  = True,
            num_workers = self.args.basic.nb_workers,
            collate_fn  = self.dataset.collate,
            shuffle     = False,
        )
    

    def scheduler_step(self, valid_loss):

        if self.args.optimizer.name == 'SGD':
            self.scheduler.step()
            lr = self.scheduler.get_lr()[0]
        elif self.args.optimizer.name == 'ADAM':
            self.scheduler.step(valid_loss)
            lr = self.optimizer.param_groups[0]['lr']
            if lr < 1e-5:
                self.console.log('=> !! learning rate is smaller than threshold !!')
        return lr


    def run(self):
        
        self.console.log(f'=> [5] Train Genotypes')
        self.lr = self.args.optimizer.lr
        self.max_metric = 0.
        self.min_metric = 1000.
        for i_epoch in range(self.args.basic.train_epochs):
            # self.model.drop_path_prob = self.args.ds.drop_path_prob * i_epoch / args.basic.epochs
            #! 训练
            train_result = self.train(i_epoch, 'train')
            self.console.log(f"[green]=> train result [{i_epoch}] - loss: {train_result['loss']:.4f} - metric : {train_result['metric']:.4f}")
            with torch.no_grad():
                if self.val_data is not None:
                    val_result   = self.infer(i_epoch, self.val_queue, 'val')
                    self.console.log(f"[yellow]=> valid result [{i_epoch}] - loss: {val_result['loss']:.4f} - metric : {val_result['metric']:.4f}")
                test_result  = self.infer(i_epoch, self.test_queue, 'test')
                self.console.log(f"[underline][red]=> test  result [{i_epoch}] - loss: {test_result['loss']:.4f} - metric : {test_result['metric']:.4f}")
                self.max_metric = max(self.max_metric, test_result['metric'])
                self.min_metric = min(self.min_metric, test_result['metric'])
                self.console.log(f"max metric: {self.max_metric:.5f}, min metric: {self.min_metric:.5f}")

                if i_epoch % self.args.visualize.interval == 0:
                    self.plot(i_epoch, self.plot_queue)
                step_loss = val_result['loss'] if self.val_data is not None else test_result['loss']
                self.lr = self.scheduler_step(step_loss)
        
        self.console.log(f'=> Finished! Genotype = {self.args.ds.load_genotypes}')
    

    @record_run('train')
    def train(self, i_epoch, stage = 'train'):

        self.model.train()
        epoch_loss   = 0
        epoch_metric = 0
        desc         = '=> training'
        device       = torch.device('cuda')

        with tqdm(self.train_queue, desc = desc, leave = False) as t:
            for i_step, (batch_graphs, batch_targets) in enumerate(t):
                #! 1. 准备训练集数据
                G = batch_graphs.to(device)
                V = batch_graphs.ndata['feat'].to(device)
                E = batch_graphs.edata['feat'].to(device)
                batch_targets = batch_targets.to(device)
                # plot_graphs_threshold(self.args, G, [E, E, E, E])
                #! 2. 优化模型参数
                self.optimizer.zero_grad()
                batch_scores = self.model({'G': G, 'V': V, 'E': E})
                loss         = self.loss_fn(batch_scores, batch_targets, graph = batch_graphs, stage = stage)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                epoch_loss   += loss.detach().item()
                epoch_metric += self.metric(batch_scores, batch_targets, graph = batch_graphs, stage = stage)

                loss_avg   = epoch_loss / (i_step + 1)
                metric_avg = epoch_metric / (i_step + 1)

                result = {'loss' : loss_avg, 'metric' : metric_avg}
                t.set_postfix(lr = self.lr, **result)
                
        return result


    @record_run('infer')
    def infer(self, i_epoch, dataloader, stage = 'infer'):

        self.model.eval()
        epoch_loss   = 0
        epoch_metric = 0
        desc         = '=> inferring'
        device       = torch.device('cuda')

        with tqdm(dataloader, desc = desc, leave = False) as t:
            for i_step, (batch_graphs, batch_targets) in enumerate(t):
                G = batch_graphs.to(device)
                V = batch_graphs.ndata['feat'].to(device)
                E = batch_graphs.edata['feat'].to(device)
                batch_targets = batch_targets.to(device)
                batch_scores  = self.model({'G': G, 'V': V, 'E': E})
                loss          = self.loss_fn(batch_scores, batch_targets, graph = batch_graphs, stage = stage)

                epoch_loss   += loss.detach().item()
                epoch_metric += self.metric(batch_scores, batch_targets, graph = batch_graphs, stage = stage)

                loss_avg   = epoch_loss / (i_step + 1)
                metric_avg = epoch_metric / (i_step + 1)

                result = {'loss' : epoch_loss / (i_step + 1), 'metric' : metric_avg}
                t.set_postfix(**result)

        return result


    def plot(self, i_epoch, dataloader):

        self.model.eval()
        device = torch.device('cuda')
        for i_step, (batch_graphs, batch_targets) in enumerate(dataloader):
            # if i_step >= self.args.visualize.examples:
            #     continue
            
            if i_step not in self.args.visualize.example_list:
                continue
            
            G = batch_graphs.to(device)
            V = batch_graphs.ndata['feat'].to(device)
            E = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            
            batch_scores  = self.model({'G': G, 'V': V, 'E': E})
            plot_graphs(
                args = self.args, 
                G = G.cpu(), 
                Vs = [cell.Vs for cell in self.model.cells],
                Es = [cell.Es for cell in self.model.cells], 
                epoch_id = i_epoch, 
                example_id = i_step, 
                V = V.cpu(), 
                L = batch_targets.cpu(), 
            )


@hydra.main(config_path = 'conf', config_name = 'defaults')
def app(args):
    OmegaConf.set_struct(args, False)
    console = Console()
    vis = Syntax(OmegaConf.to_yaml(args), "yaml", theme="monokai", line_numbers=True)
    richPanel = Panel.fit(vis)
    console.print(richPanel)
    Trainer(args).run()


if __name__ == '__main__':
    app()
