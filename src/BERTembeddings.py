'''import re
import pandas as pd
import numpy as np
import torch, torch.nn
from transformers import BertModel, BertTokenizer, pipeline
from optimum.bettertransformer import BetterTransformer
from tqdm.auto import tqdm
from .utilities import pooling_in_place
'''

import torch
from .baseEmbedding import baseEmbedding
from transformers import BertModel, BertTokenizer
from optimum.bettertransformer import BetterTransformer

class BERTembeddings(baseEmbedding):
    
    def __init__(self, 
                 type_embedding = "Rostlab/prot_bert",
                 device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 max_memory_mapping = {0: "10GB", 'cpu': "8GB"} ) -> None:
    

        super(BERTembeddings,self).__init__(  
                                            type_embedding = "facebook/esm2_t30_150M_UR50D",
                                            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                                            max_memory_mapping = {0: "10GB", 'cpu': "8GB"}
                                            )
        
        self.tokenizer = BertTokenizer.from_pretrained(type_embedding, do_lower_case=False)
        self.model = BetterTransformer.transform(
                            BertModel.from_pretrained(type_embedding, 
                                                    low_cpu_mem_usage=True, 
                                                    max_memory=max_memory_mapping#,
                                                    #output_hidden_states = True
                                            ),
                            keep_original_model=True ).to(device=self.device)

        if torch.cuda.is_available():
            self.model = self.model.half()
            #self.model = self.model.eval()

"""
class BERTembeddings():
    
    def __init__(self, 
                 type_embedding = "Rostlab/prot_bert",
                 device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 max_memory_mapping = {0: "10GB", 'cpu': "8GB"} ) -> None:
    

        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(type_embedding, do_lower_case=False)
        self.model = BetterTransformer.transform(
                            BertModel.from_pretrained(type_embedding, 
                                                    low_cpu_mem_usage=True, 
                                                    max_memory=max_memory_mapping#,
                                                    #output_hidden_states = True
                                            ),
                            keep_original_model=True ).to(device=self.device)

        if torch.cuda.is_available():
            self.model = self.model.half()
            #self.model = self.model.eval()
            
    def __get_features_from_embeddings(self, embeddings, attention_mask):
        features = [] 
        for seq_num in range(len(embeddings)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embeddings[seq_num][1:seq_len-1]
            features.append(seq_emd)

    def embed_dataset(self, dataset_seqs, path='', shift_left = 1, shift_right = -1):
        inputs_embedding = torch.tensor([]).to(self.device)

        cont=0
        
        for sample in tqdm(dataset_seqs):
            with torch.no_grad():

            
                ''' Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)'''
                #import pdb; pdb.set_trace()
                sequences_Example = sample #["A E T C Z A O","S K T Z P"]
                sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

                '''. Tokenize, encode sequences and load it into the GPU if possibile'''
                ids = self.tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)
                input_ids = torch.tensor(ids['input_ids']).to(self.device)
                attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

                ''' Extracting sequences' features and load it into the CPU if needed'''
                with torch.no_grad():
                    embedding = self.model(input_ids=input_ids,attention_mask=attention_mask)[0]


                pooled_embed = pooling_in_place(embedding)
                inputs_embedding = torch.cat([inputs_embedding, pooled_embed])

                #embedding = get_features_from_embeddings(embedding)
                '''If I want to save the embeddings in disk instead, just use this chunk of code'''
                '''------------------------------------------------------------------------------'''
                '''embedding = embedding.detach().cpu().numpy()
                np.savez(path + str(cont)+'.npz',  embedding )
                cont+=1'''
                '''------------------------------------------------------------------------------'''
        
        
        return inputs_embedding
"""        


