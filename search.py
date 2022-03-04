import os
import sys
import dgl
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from data import *
from models.model_search import *
from utils.utils import *
from searcher.darts import DARTS
from searcher.sgas import SGAS
# from architect.architect import Architect
from utils.plot_genotype import plot_genotypes
from utils.visualize import *

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import warnings
warnings.filterwarnings('ignore')
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

class Searcher(object):
    
    def __init__(self, args):

        self.args = args
        self.console = Console()

        self.console.log('=> [1] Initial settings')
        np.random.seed(args.basic.seed)
        torch.manual_seed(args.basic.seed)
        torch.cuda.manual_seed(args.basic.seed)
        cudnn.benchmark = True
        cudnn.enabled   = True

        self.console.log('=> [2] Initial models')
        self.metric    = load_metric(args)
        self.loss_fn   = get_loss_fn(args).cuda()
        self.model     = Model_Search(args, get_trans_input(args), self.loss_fn).cuda()
        self.console.log(f'[red]=> Supernet Parameters: {count_parameters_in_MB(self.model):.4f} MB')

        self.console.log(f'=> [3] Preparing dataset')
        self.dataset     = load_data(args)
        if args.ds.pos_encode > 0:
            self.console.log(f'[red]==> [3.1] Adding positional encodings')
            self.dataset._add_positional_encodings(args.ds.pos_encode)
        self.search_data = self.dataset.train
        self.val_data    = self.dataset.val
        self.test_data   = self.dataset.test
        self.load_dataloader()

        self.console.log(f'=> [4] Initial optimizer')
        self.optimizer   = torch.optim.SGD(
            params       = self.model.parameters(),
            lr           = args.optimizer.lr,
            momentum     = args.optimizer.momentum,
            weight_decay = args.optimizer.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
            optimizer  = self.optimizer,
            T_max      = float(args.basic.search_epochs),
            eta_min    = args.optimizer.lr_min
        )
        
        if args.optimizer.search_mode == 'darts':
            SEARCHER = DARTS
        elif args.optimizer.search_mode == 'sgas':
            SEARCHER = SGAS
        else:
            raise Exception("Unknown Search Mode!")

        self.searcher = SEARCHER({
            "args": self.args, 
            "model_search": self.model, 
            "arch_queue": self.arch_queue, 
            "para_queue": self.para_queue, 
            "loss_fn": self.loss_fn, 
            "metric": self.metric, 
            "optimizer": self.optimizer, 
        })



    def load_dataloader(self):

        num_search  = int(len(self.search_data) * self.args.basic.data_clip)
        indices     = list(range(num_search))
        split       = int(np.floor(self.args.basic.portion * num_search))
        self.console.log(f'=> Para set size: {split}, Arch set size: {num_search - split}')
        
        self.para_queue = torch.utils.data.DataLoader(
            dataset     = self.search_data,
            batch_size  = self.args.ds.batch,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory  = True,
            num_workers = self.args.basic.nb_workers,
            collate_fn  = self.dataset.collate
        )

        self.arch_queue = torch.utils.data.DataLoader(
            dataset     = self.search_data,
            batch_size  = self.args.ds.batch,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
            pin_memory  = True,
            num_workers = self.args.basic.nb_workers,
            collate_fn  = self.dataset.collate
        )

        if self.val_data is not None:
            num_valid = int(len(self.val_data) * self.args.basic.data_clip)
            indices   = list(range(num_valid))

            self.val_queue  = torch.utils.data.DataLoader(
                dataset     = self.val_data,
                batch_size  = self.args.ds.batch,
                sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
                pin_memory  = True,
                num_workers = self.args.basic.nb_workers,
                collate_fn  = self.dataset.collate
            )

        num_test = int(len(self.test_data) * self.args.basic.data_clip)
        indices  = list(range(num_test))

        self.test_queue = torch.utils.data.DataLoader(
            dataset     = self.test_data,
            batch_size  = self.args.ds.batch,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
            pin_memory  = True,
            num_workers = self.args.basic.nb_workers,
            collate_fn  = self.dataset.collate
        )


    def run(self):

        self.console.log(f'=> [4] Search & Train')
        for i_epoch in range(self.args.basic.search_epochs):
            self.scheduler.step()
            self.lr = self.scheduler.get_lr()[0]
            if i_epoch % self.args.visualize.report_freq == 0:
                # todo report genotype
                geno = self.searcher.genotypes()
                plot_genotypes(self.args, i_epoch, geno)
                print( geno )

            search_result = self.searcher.search({"lr": self.lr, "epoch": i_epoch})
            self.console.log(f"[green]=> [{i_epoch}] search result - loss: {search_result['loss']:.4f} - metric : {search_result['metric']:.4f}")

            with torch.no_grad():
                if self.val_data is not None:
                    val_result    = self.infer(self.val_queue)
                    self.console.log(f"[green]=> [{i_epoch}] valid result  - loss: {val_result['loss']:.4f} - metric : {val_result['metric']:.4f}")

                test_result   = self.infer(self.test_queue)
                self.console.log(f"[underline][red]=> [{i_epoch}] test  result  - loss: {test_result['loss']:.4f} - metric : {test_result['metric']:.4f}")


    def infer(self, dataloader):

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
                batch_scores  = self.model({'G': G, 'V': V, 'E': E, 'arch_topo': self.searcher.arch_topo})
                loss          = self.loss_fn(batch_scores, batch_targets)

                epoch_loss   += loss.detach().item()
                epoch_metric += self.metric(batch_scores, batch_targets)
                t.set_postfix(loss   = epoch_loss / (i_step + 1), 
                              metric = epoch_metric / (i_step + 1))

        return {'loss'   : epoch_loss / (i_step + 1), 
                'metric' : epoch_metric / (i_step + 1)}



@hydra.main(config_path = 'conf', config_name = 'defaults')
def app(args):
    OmegaConf.set_struct(args, False)
    console = Console()
    vis = Syntax(OmegaConf.to_yaml(args), "yaml", theme="monokai", line_numbers=True)
    richPanel = Panel.fit(vis)
    console.print(richPanel)

    data_path = os.path.join(get_original_cwd(), args.ds.arch_save, args.ds.data)
    Path(data_path).mkdir(parents = True, exist_ok = True)
    with open(os.path.join(data_path, "configs.txt"), "w") as f:
        f.write(str(args))

    Searcher(args).run()


if __name__ == '__main__':
    app()