"""import re
import pandas as pd
import numpy as np
import torch, torch.nn
from transformers import EsmModel, EsmTokenizer
from optimum.bettertransformer import BetterTransformer
from tqdm.auto import tqdm
from .utilities import pooling_in_place
"""
import torch
from .baseEmbedding import baseEmbedding
from transformers import EsmModel, EsmTokenizer
from optimum.bettertransformer import BetterTransformer
import esm
from tqdm.auto import tqdm

class ESM2embeddings(baseEmbedding):
    
    def __init__(self, 
                 type_embedding = "facebook/esm2_t30_150M_UR50D",
                 device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 type_tool = 'FacebookESM2',
                 max_memory_mapping = {0: "10GB", 'cpu': "8GB"} ) -> None:
    
        super(ESM2embeddings,self).__init__(  
                                            type_embedding = "facebook/esm2_t30_150M_UR50D",
                                            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                                            max_memory_mapping = {0: "10GB", 'cpu': "8GB"}
                                            )
        self.device = device
        self.type_tool=type_tool
        
        if self.type_tool=='FacebookESM2' or self.type_tool!='huggingface':
            # Load ESM-2 model
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D() #esm2_t6_8M_UR50D()
            self.model = self.model.to(self.device)
            self.tokenizer = self.alphabet.get_batch_converter()
            
        else:
            self.tokenizer = EsmTokenizer.from_pretrained(type_embedding, do_lower_case=False)
            self.model =   EsmModel.from_pretrained(type_embedding, 
                                                        low_cpu_mem_usage=True, 
                                                        max_memory=max_memory_mapping#,
                                                        #output_hidden_states = True
                                                ).to(device=self.device)

        if torch.cuda.is_available() and self.device!='cpu':
            self.model = self.model.half()
            #self.model = self.model.eval()
    
    def truncate(self, sequences, length):
        '''Function to truncate protein sequences at a given length'''
        num_truncated = len([seq for seq in sequences if len(seq) > length])
        print(num_truncated, 'sequences were too long and have been truncated to', length, 'AA')
        sequences = [seq[:length] for seq in sequences]
        
        return sequences

    def esm2embedding(self, all_data, device, truncate_length=1000, layer_index=6, pt_batch_size=1, folder_path='../prepro_embeddings/esm2_embeddings', **kargs):
        import os, ipdb
        #ipdb.set_trace()
        corresp_idxdf_file = {}
        
        if 'correspondence_idx_df_file' in kargs:
            corresp_idxdf_file = kargs['correspondence_idx_df_file']
        #embeddings = torch.tensor([]).to(device)
        #all_data.sequence = self.truncate(all_data.sequence.values, truncate_length)
        embeddings = torch.zeros(len(all_data), 1280)
        
        for i in tqdm(range(0,len(all_data), pt_batch_size)):

            batch = all_data.iloc[i:i+pt_batch_size]
            
            assert pt_batch_size==1, "Denied, you must select pt_batch_size=1 to guarantee the statistical artifact \
                                        given by mismatches on values regarding padding strategies. Unless you are \
                                        going to use fine tunning, if so, the code must be modified" 
            # To make it enable with Gustav's data or just for reading one by one instead of batches. This can be improved with decorators
            if corresp_idxdf_file: #if not kargs 
                idx = corresp_idxdf_file[batch.iloc[[0]].index.item()]
                
                if os.path.exists(f"{folder_path}/{str(idx)+'_'+ batch.iloc[0].full_name }.pt"):
                    tmp = torch.load(f"{folder_path}/{str(idx)+'_'+ batch.iloc[0].full_name }.pt")
                    tmp = tmp.mean(0).reshape(1,-1)
                    embeddings[i] = tmp
                    continue

            #ipdb.set_trace()
            esm_data = list(zip(batch.index, batch.sequence))
            batch_labels, batch_strs, batch_tokens = self.tokenizer(esm_data)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[layer_index])
            token_embeddings = results["representations"][layer_index]

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for j, tokens_len in enumerate(batch_lens):
                #torch.save(token_embeddings[j].cpu().detach(), f"{folder_path}/{str(i+j)+'_'+str(tokens_len.item())+'_'+ batch.iloc[j].full_name }.pt")
                '''saving embedding completely including start-end tokens at corners'''
                #torch.save(token_embeddings[j].cpu().detach(), f"{folder_path}/{str(i+j)+'_'+ batch.iloc[j].full_name }.pt")
                '''saving embedding without start-end tokens at corners'''
                if not kargs: #To make able the method with Gustav's data. This can be a bit more clear if I make decorators in the future
                    torch.save(token_embeddings[j, 1 : tokens_len - 1].cpu().detach(), f"{folder_path}/{str(i+j)+'_'+ batch.iloc[j].full_name }.pt")
                    
                #sequence_embeddings = token_embeddings[j].mean(0).reshape(1,-1)
                sequence_embeddings = token_embeddings[j, 1 : tokens_len - 1].mean(0).reshape(1,-1)
                #embeddings = torch.cat([embeddings, sequence_embeddings])
                embeddings[i+j] = sequence_embeddings
            
            torch.cuda.empty_cache()
                    
        return embeddings


