import torch
import esm
from tqdm.auto import tqdm

class ESM2_IF1_embeddings():
    
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.pooling_type = True
    
    
    def ESM2_IF_repr(self, df, folder_path):
        self.model.eval()
        embeddings = []#torch.zeros(len(df),512)

        with torch.no_grad():
            for i in tqdm(range(0,len(df))):
                #fpath = train.iloc[i].Structure_Alphafold THIS IS A SUPERSTUPID BUG
                fpath = df.iloc[i].Structure_Alphafold
                structure = esm.inverse_folding.util.load_structure(fpath)
                coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)

                rep = esm.inverse_folding.util.get_encoder_output(self.model, self.alphabet, coords)
                #guide_rep = get_guided_encoder_output(model, alphabet, coords, native_seq)
                torch.save(rep.cpu().detach(), f"{folder_path}/{str(i)+'_'+df.iloc[i].protein_id}.pt")
                
                embeddings.append( rep.mean(0).reshape(1,-1) )
                
        if not embeddings:
            return None
        else: 
            samples_emb = len(embeddings)
            return torch.stack(embeddings).reshape(samples_emb,-1)