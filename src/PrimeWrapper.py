import sys,os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../ref_models/Prime/"))
from ref_models.Prime.predict_ogt import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrimeBase(PrimeV0):

    @classmethod
    def load(cls, model_name, folder_path):
        if model_name == "prime-base":
            model = PrimeV0(ogt_head=True)
            checkpoint_path = folder_path+"primel-base.ckpt"
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            return model

        elif model_name == "prime-tm-fine-tuning":
            model = PrimeV0(ogt_head=False)
            checkpoint_path = folder_path+"prime-tm-fine-tuning.ckpt"
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            return model

        else:
            raise ValueError(f"Unknown model name {model_name}.")

    def __init__(self, config=None, ogt_head=False):
        super(PrimeV0,self).__init__(config, ogt_head)


@torch.no_grad()
def score_file(model, tokenizer, fasta_file):
    if isinstance(fasta_file, list):
        records = fasta_file
    else:
        records = [ str(i.seq) for i in list(SeqIO.parse(fasta_file, "fasta")) ]
    data = []
    import gc
    for sequence in tqdm(records):
        pred_ogt = predict_sequence(model, tokenizer, sequence ) #str(sequence.seq))
        #torch.cuda.empty_cache(); gc.collect()
        data.append({
            "sequence": sequence,
            "predicted_ogt": pred_ogt
            #pred_ogt
        })
    return pd.DataFrame(data)

def truncate( sequences, length):
    '''Function to truncate protein sequences at a given length'''
    num_truncated = len([seq for seq in sequences if len(seq) > length])
    print(num_truncated, 'sequences were too long and have been truncated to', length, 'AA')
    sequences = [seq[:length] for seq in sequences]
    return sequences

def main(args, out_path='../datasets/PrimeOGT_Meltome.csv', 
                        loader_path="../ref_models/Prime/models/", device=device):
    
    print("... Loading Prime OGT predictor... \n\n")
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = PrimeBase.load("prime-base", loader_path )
    model.to(device)
    model=model.half()
    print("... Predicting OGTs via Prime ... \n\n")

    if isinstance(args, list): set_prots = args 
    else:set_prots =  args.fasta
    
    set_prots = truncate(set_prots, 1500)
    df = score_file(model, tokenizer, set_prots)
    print(df)

    import ipdb; ipdb.set_trace()
    #if args.output is not None:
    df.to_csv(out_path, sep="\t", index=False)
    return df



if __name__ == '__main__':    
    def create_parser():
        psr = ArgumentParser()
        psr.add_argument("--model_name", type=str,
                         default="prime-base",
                         choices=["templ-base"],
                         help="model name")
        psr.add_argument("--fasta", type=str, default="../ref_models/Prime/datasets/OGT/ogt_small.fasta")
        psr.add_argument("--output", type=str, default="../ref_models/Prime/datasets/OGT/ogt_prediction.tsv")
        return psr


    parser = create_parser()
    cli_args = parser.parse_args()
    main(cli_args)