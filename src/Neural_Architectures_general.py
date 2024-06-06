
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, input):
        return input.view(input.size(0), -1) 

'''
    IMPORTANT FACT: LIGHT ATTENTION KERNEL SIZE MATTERS!!!!!!
    - BE CAREFUL, WHAT I HAVE SEEN IS THAT THE BIGGER KERNEL SIZE GET BETTER SPEARMAN CORRELATION IN TERMS OF PREDICTIONS
    - IN THIS CASE, THE KERNEL SIZE AFFECTS DIRECTLY THE PERFORMANCE BETWEEN USING ESM2 ALONG VS ESM2+ INVERSE FOLDING 
    (MORE KERNEL, BETTTER RESULTS)
'''
class LightAttention(nn.Module):
    # borrowed from HannesStark repo https://github.com/HannesStark/protein-localization.git

    def __init__(self, embeddings_dim=1024, output_dim=1, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, output_dim)

#    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        #import ipdb; ipdb.set_trace()
        o = self.feature_convolution(x.permute(0,2,1))  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x.permute(0,2,1))  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        '''attention = attention.masked_fill(mask[:, None, :] == False, -1e9)'''

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]


class LA_MLP_pred(nn.Module):
    
    def __init__(self, embeddings_dim=1024, output_dim=2, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25, device: str='cuda',
                 output_LA:int =128):
        super(LA_MLP_pred, self).__init__()
        self.LAmodule = LightAttention(embeddings_dim=embeddings_dim, 
                                       output_dim=output_LA, 
                                       dropout=dropout, 
                                       kernel_size=kernel_size, 
                                       conv_dropout=conv_dropout).to(device)
        
        self.FFNN = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(output_LA, 128),
            torch.nn.LeakyReLU(0.1), #nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(0.1), #nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        ).to(device)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        #import ipdb; ipdb.set_trace()
        x = self.LAmodule(x)
        x = self.FFNN(x)
        return x
    
def la_mlp(**kwargs):
    dim=kwargs['dim']
    device = kwargs['device']
    output_dim = kwargs['output_dim']

    return LA_MLP_pred(dim, 
                        output_dim=output_dim, 
                        dropout=0.25, 
                        kernel_size=9, 
                        conv_dropout= 0.25,
                        device=device)

model_mode_dict = {
    'Encoder_LA_MLP_Single': [la_mlp, 'single_opt', 1],
    'Encoder_LA_MLP_Single_Bias_Organism': [la_mlp, 'single_opt', 2],
    'Encoder_LA_MLP_Rank_N_Contrast': [la_mlp, 'dual_opt',1]
}


class Encoder_Manager(nn.Module):
    def __init__(self, name='Encoder_LA_MLP_Single', dim_embedding=1280, device='cuda'):
        super(Encoder_Manager, self).__init__()
        model_fun, self.opt_mode, out_layer = model_mode_dict[name]
        self.encoder = model_fun(dim=dim_embedding, device=device, output_dim=out_layer)
        self.to(device)

    def set_optimizer(self):
        #import ipdb; ipdb.set_trace()
        if self.opt_mode == 'single_opt':
            # THIS LEARNING RATE IS SPECIFIED JUST FOR GLOBAL MELTOME FLIP EXPERIMENT.
            # IN CASE OF RUNNING SPECIFIC SPECIES EXPERIMENT, IT IS REQUIRED TO INCREASE
            # THE LEARNING RATE UP TO 1E-3.
            #return {'predictor_opt':torch.optim.AdamW(self.encoder.parameters(), lr=1e-4)}
            return {'predictor_opt':torch.optim.AdamW(self.encoder.parameters(), lr=1e-3)}
        else:
            return  {
                    'Representation_opt':torch.optim.AdamW(self.encoder.LAmodule.parameters(), lr=1e-5), 
                    'predictor_opt':torch.optim.AdamW(self.encoder.FFNN.parameters(), lr=1e-3)}
                    #'predictor_opt':torch.optim.AdamW(self.encoder.FFNN.parameters(), lr=1e-4)}        
        
    
    # To Freeze modules, enabling_tag=False, otherwise enabling_tag=True
    def __freeze_unfreeze_target(self, target, enabling_tag= False):
        #import ipdb; ipdb.set_trace()
        for param in target.parameters():
            param.requires_grad = enabling_tag
        #print( [i.requires_grad for i in target.parameters()] )
    
    def setup_tunning(self, mode='both'):
        self.model_mode=mode
        if mode=='representation':
            print('Starting representation block setup')
            self.__freeze_unfreeze_target(self.encoder.LAmodule, enabling_tag=True)
            self.__freeze_unfreeze_target(self.encoder.FFNN, enabling_tag=False)
        elif mode=='predictor':
            print('Starting regresor block setup')
            self.__freeze_unfreeze_target(self.encoder.LAmodule, enabling_tag=False)
            self.__freeze_unfreeze_target(self.encoder.FFNN, enabling_tag=True)
        #else:
        #    print('Starting full block setup')
        #    self.__freeze_unfreeze_target(self.encoder.LAmodule, enabling_tag=True)
        #    self.__freeze_unfreeze_target(self.encoder.FFNN, enabling_tag=True)

    def forward(self, x):
        if self.model_mode=='representation':
            return self.encoder.LAmodule(x)
        else:
            return self.encoder(x)
            
