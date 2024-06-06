import json, time, os, sys, glob
import numpy as np

import logging
import pickle
import json
import torch
import os.path as osp


import warnings
warnings.filterwarnings('ignore')

import sys,os
sys.path.append(os.path.abspath("../ref_models/PiFold"))

from ref_models.PiFold.main import Exp
from ref_models.PiFold.methods import ProDesign
from ref_models.PiFold.API import Recorder
from ref_models.PiFold.utils import *
from ref_models.PiFold.methods.utils import cuda
from ref_models.PiFold.parser import create_parser
from . import Structure_utils as Structure_utils

import torch.nn.functional as F
from ref_models.PiFold.API.dataloader_gtrans import featurize_GTrans
from tqdm.auto import tqdm


def get_parser():
  import argparse
  parser = argparse.ArgumentParser()
  # Set-up parameters
  parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
  parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
  parser.add_argument('--res_dir', default='results', type=str)
  parser.add_argument('--ex_name', default='ProDesign', type=str)
  parser.add_argument('--use_gpu', default=True, type=bool)
  parser.add_argument('--gpu', default=0, type=int)
  parser.add_argument('--seed', default=111, type=int)

  # CATH
  # dataset parameters
  parser.add_argument('--data_name', default='CATH', choices=['CATH', 'TS50'])
  parser.add_argument('--data_root', default='data/')
  parser.add_argument('--batch_size', default=8, type=int)
  parser.add_argument('--num_workers', default=8, type=int)

  # method parameters
  parser.add_argument('--method', default='ProDesign', choices=['ProDesign'])
  parser.add_argument('--config_file', '-c', default=None, type=str)
  parser.add_argument('--hidden_dim',  default=128, type=int)
  parser.add_argument('--node_features',  default=128, type=int)
  parser.add_argument('--edge_features',  default=128, type=int)
  parser.add_argument('--k_neighbors',  default=30, type=int)
  parser.add_argument('--dropout',  default=0.1, type=int)
  parser.add_argument('--num_encoder_layers', default=10, type=int)

  # Training parameters
  parser.add_argument('--epoch', default=100, type=int, help='end epoch')
  parser.add_argument('--log_step', default=1, type=int)
  parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
  parser.add_argument('--patience', default=100, type=int)

  # ProDesign parameters
  parser.add_argument('--updating_edges', default=4, type=int)
  parser.add_argument('--node_dist', default=1, type=int)
  parser.add_argument('--node_angle', default=1, type=int)
  parser.add_argument('--node_direct', default=1, type=int)
  parser.add_argument('--edge_dist', default=1, type=int)
  parser.add_argument('--edge_angle', default=1, type=int)
  parser.add_argument('--edge_direct', default=1, type=int)
  parser.add_argument('--virtual_num', default=3, type=int)
  args = parser.parse_args([])
  return args

from ref_models.PiFold.methods.prodesign_model import ProDesign_Model

class PiFold_embedder(ProDesign_Model):
    
    def __init__(self, args, **kwargs):
        super().__init__(args,  **kwargs)
        
    def forward(self, h_V, h_P, P_idx, batch_id, S=None, AT_test = False, mask_bw = None, mask_fw = None, decoding_order= None, return_logit=False):
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))
        
        h_V, h_P = self.encoder(h_V, h_P, P_idx, batch_id)

        return h_V, h_P
        

class PiFold_setup(Exp):
    
    def __init__(self, args, show_params=True):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        if show_params:
            print_log(output_namespace(self.args))
    
    
    def _preparation(self):
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        
        #self._get_data()
        self._build_method()

    def _build_method(self):
        if self.args.method == 'ProDesign':
            #self.method = ProDesign(self.args, self.device, steps_per_epoch)
            self.method = PiFold_embedder(self.args).to(self.device)

    def _embedding_data(self, PDB_path):

        data = Structure_utils.parsePDB('{}'.format(PDB_path))

        with torch.no_grad():
            alphabet='ACDEFGHIKLMNPQRSTVWY'
            batch = featurize_GTrans([data])
            #batch = featurize_GTrans([data])
            X, S, score, mask, lengths = cuda(batch, device = self.device)
            X, S, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = self.method._get_features(S, score, X=X, mask=mask)
            NodeEmb, EdgeEmb  = self.method(h_V, h_E, E_idx, batch_id, return_logit = True)
            return NodeEmb
            
    
    def IF_repr(self, df, folder_path, **kargs):
        self.method.eval()
        embeddings = []#torch.zeros(len(df),512)
        #import ipdb; ipdb.set_trace()
        
        corresp_idxdf_file = {}
        if 'correspondence_idx_df_file' in kargs:
            corresp_idxdf_file = kargs['correspondence_idx_df_file']

        with torch.no_grad():
            for i in tqdm(range(0,len(df))):
                #fpath = train.iloc[i].Structure_Alphafold THIS IS A SUPERSTUPID BUG
                #if corresp_idxdf_file: #if not kargs 
                '''SEE IF IT IS WORKING FINE'''
                idx = ( i, corresp_idxdf_file[df.iloc[[i]].index.item()] )[ bool(corresp_idxdf_file)]
                #else:
                    
                if os.path.exists(f"{folder_path}/{str(idx)+'_'+df.iloc[i].protein_id}.pt"):
                    tmp = torch.load(f"{folder_path}/{str(idx)+'_'+df.iloc[i].protein_id}.pt")
                    embeddings.append( tmp.mean(0).reshape(1,-1))
                else:
                    fpath = df.iloc[i].Structure_Alphafold
                    
                    GraphEmb = self._embedding_data(fpath)

                    #guide_rep = get_guided_encoder_output(model, alphabet, coords, native_seq)
                    if not kargs:
                        torch.save(GraphEmb.cpu().detach(), f"{folder_path}/{str(i)+'_'+df.iloc[i].protein_id}.pt")
                
                    embeddings.append( GraphEmb.mean(0).reshape(1,-1) )

                
        if not embeddings:
            return None
        else: 
            samples_emb = len(embeddings)
            return torch.stack(embeddings).reshape(samples_emb,-1)
  