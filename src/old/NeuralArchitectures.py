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

class regressionHead(torch.nn.Module):

    def __init__(self, shape_embedding):
        super(regressionHead, self).__init__()
        self.shape_emb = np.prod(shape_embedding)
        #self.input_shape = input_shape # pos0 = #channels, pos1 = #diagonal comps, or viseversa


        self.FFNN = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(self.shape_emb, 512),
            torch.nn.Sigmoid(), #nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.Sigmoid(), #nn.ReLU(),
            torch.nn.Linear(512, 1),
        )
        
    def forward(self, x):
        z = self.FFNN(x)
        return z
    
class CNNpooling(torch.nn.Module):
    
    def __init__(self, dim_seq, dim_embed):
        super(CNNpooling, self).__init__()
        self.channel_emb = dim_embed
        self.seq_dim = dim_seq
        kernel=1
        
        self.cnn_pooling = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.channel_emb, out_channels=self.channel_emb, kernel_size=kernel, stride=1, padding=kernel//2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channel_emb, out_channels=1, kernel_size=kernel, stride=1, padding=kernel//2),
        )
    
    def forward(self,x):
        x = self.cnn_pooling(x)
        return x
    
    
class tunning_pool_and_FFNN(torch.nn.Module):
    
    def __init__(self, dim_seq, dim_embed):
        super(tunning_pool_and_FFNN, self).__init__()
        self.learned_pooling = CNNpooling(dim_seq, dim_embed)
        self.FFNN = regressionHead([dim_seq, 1])
    
    def forward(self,x):
        x = self.learned_pooling(x.permute(0,2,1))
        x = self.FFNN(x)
        return x
    

class tunning_pool_pred_byComp(torch.nn.Module):
    
    def __init__(self, dim_embed, kernel=1):
        super(tunning_pool_pred_byComp, self).__init__()
        self.channel_emb = dim_embed
        self.learned_pooling = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.channel_emb, out_channels=self.channel_emb, kernel_size=kernel, stride=1, padding=kernel//2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(in_channels=self.channel_emb, out_channels=self.channel_emb, kernel_size=kernel, stride=1, padding=kernel//2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(in_channels=self.channel_emb, out_channels=1, kernel_size=kernel, stride=1, padding=kernel//2),
        )
    
    def forward(self,x):
        x = self.learned_pooling(x.permute(0,2,1))
        #x = torch.sum(x, dim=-1)
        x = torch.mean(x, dim=-1)
        return x
        

class tunning_pool_pred_byComp2(torch.nn.Module):
    
    def __init__(self, dim_embed, kernel=1):
        super(tunning_pool_pred_byComp2, self).__init__()
        self.channel_emb = dim_embed
        self.learned_pooling = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.channel_emb, out_channels=300, kernel_size=kernel, stride=1, padding=kernel//2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(in_channels=300, out_channels=10, kernel_size=kernel, stride=1, padding=kernel//2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(in_channels=10, out_channels=1, kernel_size=3, stride=1, padding=kernel//2),
        )
    
    def forward(self,x):
        x = self.learned_pooling(x.permute(0,2,1))
        #x = torch.sum(x, dim=-1)
        x = torch.mean(x, dim=-1)
        return x
    
class tunning_pool_pred_byComp3(torch.nn.Module):
    
    def __init__(self, dim_embed, kernel=9):
        super(tunning_pool_pred_byComp3, self).__init__()
        self.channel_emb = dim_embed
        self.learned_pooling = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.channel_emb, out_channels=300, kernel_size=kernel, stride=1, padding=kernel//2),
            torch.nn.Sigmoid(),
            torch.nn.Conv1d(in_channels=300, out_channels=100, kernel_size=kernel, stride=1, padding=kernel//2),
            torch.nn.Sigmoid(),
            #torch.nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=kernel//2),
            #torch.nn.Sigmoid()
        )
        
        self.ffnn = regressionHead(shape_embedding = 100)
    
    def forward(self,x):
        #import ipdb; ipdb.set_trace()
        x = self.learned_pooling(x.permute(0,2,1))
        #x = x.sum(-1)
        x = torch.mean(x, dim=-1)
        #x = torch.sum(x, dim=-1)
        x = self.ffnn(x)
        return x


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
    
class LightAttention_noBN(nn.Module):
    # borrowed from HannesStark repo https://github.com/HannesStark/protein-localization.git

    def __init__(self, embeddings_dim=1024, output_dim=1, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention_noBN, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)


        self.output =  self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )#nn.Linear(32, output_dim)

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
        return self.output(o)  # [batchsize, output_dim]
    

class LA_MLP(nn.Module):
    
    def __init__(self, embeddings_dim=1024, output_dim=2, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25, device: str='cuda'):
        super(LA_MLP, self).__init__()
        self.LAmodule = LightAttention(embeddings_dim=embeddings_dim, output_dim=128, dropout=dropout, kernel_size=kernel_size, conv_dropout=conv_dropout).to(device)
        
        self.FFNN = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(128, 128),
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
    
    
class LA_MLP_TaskVector(nn.Module):
    
    def __init__(self, embeddings_dim=1024, 
                 output_dim=2, 
                 dropout=0.25, 
                 kernel_size=9, 
                 conv_dropout: float = 0.25, 
                 device: str='cuda', output_LA=1700, 
                 path_pretrained='', df=None):

        super(LA_MLP_TaskVector, self).__init__()
        
        #import ipdb; ipdb.set_trace()
        self.pretrained_Avg = None
        self.enable_pretrained_avg = False
        self.LAmodule = LightAttention(embeddings_dim=embeddings_dim, 
                                       output_dim=output_LA, 
                                       dropout=dropout, 
                                       kernel_size=kernel_size, 
                                       conv_dropout=conv_dropout).to(device)
    
        self.FFNN = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(output_LA, 1280),
            torch.nn.LeakyReLU(0.1), #nn.ReLU(),
            torch.nn.Linear(1280, 640),
            torch.nn.LeakyReLU(0.1), #nn.ReLU(),
            torch.nn.Linear(640, output_dim),
        ).to(device)

        #import ipdb; ipdb.set_trace()
        if len(path_pretrained)!=0 and self.is_non_empty_dir(path_pretrained):
            self.enable_pretrained_avg = True
            params = ( embeddings_dim, 
                      output_dim, 
                      dropout, 
                      kernel_size, 
                      conv_dropout, 
                      device, output_LA, ''                
                )
            
            self.LAmodule, _ ,\
                self.pretrained_Avg = self.load(path_pretrained, params)
            
            # Freeze the parameters to don't get into the optimization,
            # to act as a pretrained weights following the 
            self.pretrained_Avg.requires_grad_(False)
            for p in self.pretrained_Avg.parameters(): p.requires_grad=False
        
    #@torch.no_grad()
    def get_embeddings_Avg_species(self, x, y, df):
        list_species = df.organism.unique().tolist()
        
        """ SANITY CHECK TO SEE IF THE OPERATION IS FINE """
        #pretrained_avg_emb_species = self.pretrained_Avg(y)
        for i in list_species:
            
            mean_species = y[ \
                        df[df.organism==i].embedding_poss.tolist() \
                        ].mean(0).unsqueeze(0)
            
            idx_spe=df[df.organism==i].embedding_poss.tolist()
            x[idx_spe] = x[idx_spe] - mean_species
        
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        #import ipdb; ipdb.set_trace()
        if 'df' in kwargs and self.enable_pretrained_avg == True:
            df = kwargs['df']
            with torch.no_grad():
                task_Avg = self.pretrained_Avg(x)
            x = self.LAmodule(x)
            x = self.get_embeddings_Avg_species(x, task_Avg, df)
        else:
            x = self.LAmodule(x)
            
        x = self.FFNN(x)
        return x
    
    
    @classmethod
    def load(cls, folder_path, params, device='cuda'):
        #import ipdb; ipdb.set_trace()
        import copy
        checkpoint_path = os.listdir(folder_path)[-1]
        checkpoint = torch.load(folder_path+checkpoint_path)
        
        model = LA_MLP_TaskVector(*params)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model.LAmodule, model.FFNN, copy.deepcopy(model.LAmodule)

    def is_non_empty_dir(self, dir_name: str) -> bool:
        """
        Returns True if the directory exists and contains item(s) else False
        """
        try:
            if any(os.scandir(dir_name)):
                return True
        except (NotADirectoryError, FileNotFoundError):
            pass
        return False
         
'''------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------------------------------------------'''

class LA_MLP_TaskVectorNoBN(LA_MLP_TaskVector):

    def __init__(self, embeddings_dim=1024, 
                 output_dim=2, 
                 dropout=0.25, 
                 kernel_size=9, 
                 conv_dropout: float = 0.25, 
                 device: str='cuda', output_LA=1700, 
                 path_pretrained='', df=None):

        #import ipdb; ipdb.set_trace()
        LA_MLP_TaskVector.__init__(self,embeddings_dim, 
                                                output_dim, 
                                                dropout, 
                                                kernel_size, 
                                                conv_dropout, 
                                                device, 
                                                output_LA, 
                                                path_pretrained='', df=None)
        
 

        self.LAmodule = LightAttention_noBN(embeddings_dim=embeddings_dim, 
                                       output_dim=output_LA, 
                                       dropout=dropout, 
                                       kernel_size=kernel_size, 
                                       conv_dropout=conv_dropout).to(device)
    

        #import ipdb; ipdb.set_trace()
        if len(path_pretrained)!=0 and self.is_non_empty_dir(path_pretrained):
            self.enable_pretrained_avg = True
            params = ( embeddings_dim, 
                      output_dim, 
                      dropout, 
                      kernel_size, 
                      conv_dropout, 
                      device, output_LA, ''                
                )
            #import ipdb; ipdb.set_trace()
            self.LAmodule, _ ,\
                self.pretrained_Avg = self.load(path_pretrained, params)
            
            # Freeze the parameters to don't get into the optimization,
            # to act as a pretrained weights following the             
            #self.pretrained_Avg.weight.requires_grad_(False)
            #self.pretrained_Avg.bias.requires_grad_(False)
            for p in self.pretrained_Avg.parameters(): 
                #p.weight.requires_grad_(False)
                #p.bias.requires_grad_(False)
                p.requires_grad=False
            

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        #import ipdb; ipdb.set_trace()
        if 'df' in kwargs and self.enable_pretrained_avg == True:
            df = kwargs['df']
            
            task_Avg = self.pretrained_Avg(x)
            x = self.LAmodule(x)
            x = self.get_embeddings_Avg_species(x, task_Avg, df)
        else:
            x = self.LAmodule(x)
            
        x = self.FFNN(x)
        return x
    
    
    @classmethod
    def load(cls, folder_path, params, device='cuda'):
        #import ipdb; ipdb.set_trace()
        import copy
        checkpoint_path = os.listdir(folder_path)[-1]
        checkpoint = torch.load(folder_path+checkpoint_path)
        
        model = LA_MLP_TaskVectorNoBN(*params)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model.LAmodule, model.FFNN, copy.deepcopy(model.LAmodule)

'''------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------------------------------------------'''




        
        
''' LA_MLP_AvgTmOrg_extra is the best that works so far, but would be nice if we dont depend on the average tm per organism'''
class LA_MLP_AvgTmOrg_extra(nn.Module):
    
    def __init__(self, embeddings_dim=1024, output_dim=2, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25, device: str='cuda'):
        super(LA_MLP_AvgTmOrg_extra, self).__init__()
        self.LAmodule = LightAttention(embeddings_dim=embeddings_dim, output_dim=128, dropout=dropout, kernel_size=kernel_size, conv_dropout=conv_dropout).to(device)
        
        self.FFNN = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(128+1, 128+1),
            torch.nn.LeakyReLU(0.1), #nn.ReLU(),
            torch.nn.Linear(128+1, 64),
            torch.nn.LeakyReLU(0.1), #nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        ).to(device)
        
    def forward(self, x: torch.Tensor, avt_tmp: torch.Tensor, **kwargs) -> torch.Tensor:
        #import ipdb; ipdb.set_trace()
        x = self.LAmodule(x)
        x = torch.hstack((x,avt_tmp))
        x = self.FFNN(x)
        return x


class LA_MLP_AvgTaskVect_extra(LA_MLP_AvgTmOrg_extra):
    
    def __init__(self, embeddings_dim=1024, output_dim=2, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25, device: str='cuda', path: str=''):
        #LA_MLP_AvgTmOrg_extra.__init__(self,embeddings_dim, output_dim, dropout, kernel_size, conv_dropout, device)
        #self.LAmodule = LightAttention(embeddings_dim=embeddings_dim, output_dim=128, dropout=dropout, kernel_size=kernel_size, conv_dropout=conv_dropout).to(device)
        super(LA_MLP_AvgTaskVect_extra, self).__init__()
        #import ipdb; ipdb.set_trace()
        self.LAmodule = LA_MLP_AvgTaskVect_extra.load_pretrained_LA(
                        [embeddings_dim, output_dim, dropout, kernel_size, conv_dropout, device], 
                        path).to(device)
        
        self.FFNN = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(128, 1280),
            torch.nn.LeakyReLU(0.1), #nn.ReLU(),
            torch.nn.Linear(1280, 640),
            torch.nn.LeakyReLU(0.1), #nn.ReLU(),
            torch.nn.Linear(640, output_dim),
        ).to(device)
        
    @classmethod
    def load_pretrained_LA(cls, params, PATH: str=''):
        import os
        tmp = LA_MLP_AvgTmOrg_extra(*params)
        onlyfiles = os.listdir(PATH)
        bestepoch = onlyfiles[-1]
        checkpoint = torch.load(PATH+bestepoch)
        tmp.load_state_dict(checkpoint['model_state_dict'])
        return tmp.LAmodule
        
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        #import ipdb; ipdb.set_trace()
        x = self.LAmodule(x)
        #y = x.clone()
        
        # finding of mean of embeddings belonging to specific organism
        if 'df' in kwargs:
            df= kwargs['df']
            list_species = df.organism.unique().tolist()
            """
            mean_emb_organism = list( map(
                lambda v: x[ df[df.organism==v].embedding_poss.tolist() ].
                                                            mean(0).unsqueeze(0), 
                list_species
                ))
            mean_emb_organism_dict = dict(zip(list_species, mean_emb_organism))
            
            for i in mean_emb_organism_dict.keys():
                idx_spe=df[df.organism==i].embedding_poss.tolist()
                x[idx_spe] = x[idx_spe] - mean_emb_organism_dict[i]
            """
            
            """ SANITY CHECK TO SEE IF THE OPERATION IS FINE """
            for i in list_species:
                mean_species = x[ df[df.organism==i].embedding_poss.tolist() ].mean(0).unsqueeze(0)
                idx_spe=df[df.organism==i].embedding_poss.tolist()
                x[idx_spe] = x[idx_spe] - mean_species
                

        x = self.FFNN(x)
        return x





class LA_MLP_Geometric(nn.Module):
    
    def __init__(self, embeddings_dim=1024, output_dim=2, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25, device: str='cuda'):
        super(LA_MLP_Geometric, self).__init__()
        self.LAmodule = LightAttention(embeddings_dim=embeddings_dim, output_dim=128, dropout=dropout, kernel_size=kernel_size, conv_dropout=conv_dropout).to(device)
        
        self.FFNN = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(0.1), #nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(0.1), #nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        ).to(device)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        #import ipdb; ipdb.set_trace()
        LA_emb = self.LAmodule(x)
        x = self.FFNN(LA_emb)
        return x, LA_emb

class seq_struct_tunning_pool_and_FFNN(tunning_pool_and_FFNN):
    
    def __init__(self, dim_seq, dim_embed, dim_embed_ESM):
        super(tunning_pool_and_FFNN, self).__init__()
        self.learned_pooling = CNNpooling(dim_seq, dim_embed)
        self.flattener = Flatten()
        self.FFNN = regressionHead([dim_seq + dim_embed_ESM, 1])
        
    def forward(self, x, y):
        y = self.learned_pooling(y.permute(0,2,1))
        y = self.flattener(y)
        z = torch.cat([x,y],dim=1)
        z = self.FFNN(z)
        return z