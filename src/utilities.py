
import pandas as pd
import numpy as np
import torch, os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os.path
from os import path
from Bio.SeqIO.FastaIO import SimpleFastaParser
import seaborn as sns


def cleanup_by_organism_and_correspondence_idx(df, organism, keyword=''):
    #import ipdb; ipdb.set_trace()
    corresp_idxdf_idxfile = dict(
                                zip(
                                    df.index.values.tolist(),
                                    list(range(0,df.shape[0])
                                    )
                                    ) )
    try:
        if isinstance(organism, list):
            query_org = '|'.join(organism)
            df = df[ df['full_name'].str.contains(query_org)]
        else:
            query_org = (organism+'_'+keyword, organism)[keyword=='']
            df = df[ df['full_name'].str.contains(query_org)]
    except:
        return corresp_idxdf_idxfile, None
    
    return corresp_idxdf_idxfile, df

def read_and_plot_loss_curves(path):
    train_loss, val_loss = parse_train_progress_files(path)
    plot_train_val_loss_curves(train_loss, val_loss)

def parse_train_progress_files(path):
    train_loss=[]
    val_loss=[]
    with open(path) as f:
        for line in f:
            train_loss.append( float(line.split("loss: ")[1].split(' ')[0]) )
            val_loss.append( float(line.split("loss: ")[2].split(' ')[0]) )
    return train_loss, val_loss

def plot_train_val_loss_curves(train_loss, val_loss):
        epochs_domain = range(0, len(train_loss))
        plt.plot(epochs_domain, train_loss, 'b', label='Train Loss')
        plt.plot(epochs_domain, val_loss, 'r', label='Val Loss')
        plt.title('Train ad Evaluation Losss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Evolution')
        plt.legend(), plt.show()

def train_test_validation_splits(df):
    #length including CLS and SEP tokens for embeddings
    df['length_w_tokens']=df.sequence.str.len()+2
    test,train_tot = df.loc[df['set']=='test'],df.loc[df['set']=='train']
    #global calculation OGT as average of tm in train and validation test
    #train_tot['OGTest_organism']=[train_tot.tm_organism.mean()]*len(train_tot)
    train, val = train_tot.loc[train_tot['validation']!=True], train_tot.loc[train_tot['validation']==True]
    return train,val, test

def seek_UniprotID_association_Meltome_prots(df, df_db):
    #import ipdb; ipdb.set_trace()
    intersection = pd.merge(df.astype(str), df_db.astype(str), on=['sequence','target'],how='inner')
    intersection['full_name']=intersection.protein_id
    intersection.protein_id = intersection.protein_id.apply(lambda x: x.split('_')[0] )
    intersection['organism']= intersection.full_name.apply(lambda x: '_'.join(x.split('_')[1::]) )
    '''SEE THIS CARREFULLY, SEEMS LIKE I AM ADDING INFO FROM TEST SET WITHOUT INTENTION, COMPARE'''
    intersection['tm_organism'] = intersection.organism.map( intersection.groupby('organism').apply(lambda a: a.target.astype(float).mean()) )
    intersection.validation = intersection.validation.astype(object).apply(lambda a: True if a=='True' else a==np.nan)
    return intersection


''' Method for creating batches from a list of objects, useful for making the training by batches.
    it was necessary to do so, because the BERT embeddings just give as outputs a list of elements'''
def build_batch_iterator_sequences(sequences_total, batch_size):

    long_list = sequences_total
    sub_list_length = batch_size
    sub_lists = [
        long_list[i : i + sub_list_length]
        for i in range(0, len(long_list), sub_list_length)
    ]
    return sub_lists

def prepare_train_test_val_seqs_by_batches(train, test, val, batch_size=16):
    batch_train = build_batch_iterator_sequences(train, batch_size)
    batch_test = build_batch_iterator_sequences(test, batch_size)
    batch_val = build_batch_iterator_sequences(val, batch_size)
    return batch_train, batch_test, batch_val


def pooling_in_place(batchtensor):
  batchtensor = batchtensor.permute(0,2,1)
  GlobalAvgpooling = torch.nn.AvgPool1d(kernel_size = batchtensor.size(2) , stride = 1, padding = 0)
  pooled_seq = GlobalAvgpooling(batchtensor)
  return pooled_seq


def pooling_and_final_representation(path):
  # Due to npz compression on each embedded sequence
  sample = np.load(path)
  embedded_seq = torch.from_numpy( sample[sample.files[0]] ).to(torch.float32)
  pooled_seq = pooling_in_place(embedded_seq)

  return pooled_seq 

#the test
def pooled_set_of_sequences(folder_path):
  #import pdb; pdb.set_trace()
  list_embeddings = []
  iter_paths = os.listdir(folder_path)
  for path in tqdm(iter_paths):
    list_embeddings.append( pooling_and_final_representation(folder_path + path) )

  #import pdb;pdb.set_trace()
  representation = torch.stack(list_embeddings)
  representation = representation.view(representation.size(0),-1)
  return representation


def tensor2dataloader(tensor_data, tensor_target, batch_size=50, shuffle=True) -> torch.utils.data.DataLoader :
    #import ipdb; ipdb.set_trace()
    Dataset = torch.utils.data.TensorDataset(tensor_data, tensor_target )
    Data_Loader = torch.utils.data.DataLoader(Dataset, batch_size=batch_size, shuffle=shuffle)
    return Data_Loader


def plot_results( outcome, test_labels):

    indexes = list(range(0,len(test_labels) ) ) 

    plt.figure(figsize=(25,8))
    plt.plot(indexes, outcome.flatten().detach().cpu().tolist(), c='red', label="predicted")
    plt.plot( indexes, test_labels.flatten().detach().cpu().tolist(), c='green', label="ground truth")
    plt.legend(loc="upper left")
    #plt.ylim(-1.5, 2.0)
    plt.show()
    #import ipdb; ipdb.set_trace()
    plot_scatter_pred(outcome.flatten().detach().cpu().numpy(), 
                      test_labels.flatten().detach().cpu().numpy())
    
    

def plot_scatter_pred(predicted_values, ground_truth_values, threshold = 5.):

    # Set the style

    sns.set(style="ticks", font_scale=1.2)
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))

    # Scatter plot in linear scale
    sns.scatterplot(x=predicted_values, y=ground_truth_values, color='steelblue', alpha=0.7, s=20, ax=ax1)
    ax1.axline((0, 0), slope=1, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Predicted Values', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Ground Truth Values', fontsize=14, fontweight='bold')
    ax1.set_title('Predicted vs Ground Truth (Linear Scale)', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', which='both', length=0)
    ax1.grid(True, linestyle='--', alpha=0.3)
    sns.despine(ax=ax1)
    max_value = max(np.max(predicted_values), np.max(ground_truth_values))
    ax1.set_xlim(0, max_value)
    ax1.set_ylim(0, max_value)
    ax1.legend(['Baseline'], loc='upper left', frameon=False, fontsize=12)
    ax1.set_aspect('equal', adjustable='box')

    # Scatter plot in log scale
    sns.scatterplot(x=predicted_values, y=ground_truth_values, color='steelblue', alpha=0.7, s=20, ax=ax2)
    ax2.axline((0, 0), slope=1, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Predicted Values (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Ground Truth Values (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_title('Predicted vs Ground Truth (Log Scale)', fontsize=16, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.3)
    sns.despine(ax=ax2)
    ax2.set_xlim(0.1, 10 * max_value)
    ax2.set_ylim(0.1, 10 * max_value)
    ax2.legend(['Baseline'], loc='upper left', frameon=False, fontsize=12)
    ax2.set_aspect('equal', adjustable='box')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Define threshold for almost equal values
    #threshold = 10.

    # Color points where predicted values are almost equal to ground truth values
    diff = np.abs(predicted_values - ground_truth_values)
    colors = np.where(diff <= threshold, diff, np.nan)
    cmap = sns.color_palette("viridis", as_cmap=True)
    sc = ax1.scatter(predicted_values, ground_truth_values, c=colors, cmap=cmap, alpha=0.7, s=20)
    ax2.scatter(predicted_values, ground_truth_values, c=colors, cmap=cmap, alpha=0.7, s=20)

    # Add a colorbar for the range of colors
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Difference', fontsize=12)

    # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Display the plot
    plt.show()

    
def create_folder_embeddings_savings(tmpfiles):
    train_tmp_folder = tmpfiles+'train_embeddinds/'
    test_tmp_folder = tmpfiles+'test_embeddinds/'
    val_tmp_folder = tmpfiles+'val_embeddinds/'

    import os.path
    from os import path

    if path.exists(tmpfiles) == False:
        os.mkdir(tmpfiles)
        os.mkdir(train_tmp_folder )
        os.mkdir(test_tmp_folder )
        os.mkdir(val_tmp_folder )
    else:
        if path.exists(train_tmp_folder) == False or path.exists(test_tmp_folder) == False or path.exists(val_tmp_folder) == False:
            try:
                os.mkdir(train_tmp_folder )
                os.mkdir(test_tmp_folder )
                os.mkdir(val_tmp_folder )
            except Exception:
                pass
            

def load_full_meltome_FLIP_db(complete_meltome_db):
    with open(complete_meltome_db) as fasta_file:  # Will close handle cleanly
        identifiers = []
        sequences = []
        OGT = []
        for title, sequence in SimpleFastaParser(fasta_file):
            identifiers.append(title.split(None, 1)[0])  # First word is ID
            sequences.append(sequence)
            OGT.append( title.split(None,1)[1].split('=')[1] )
            
    full_meltome_db = pd.DataFrame( list(zip(identifiers, sequences, OGT)), columns=['protein_id', 'sequence', 'target'])
    return full_meltome_db





def get_url(url, **kwargs):
    response = requests.get(url,**kwargs)
      
    if not response.ok:
        #print(response.text)
        response.raise_for_status()
        #sys.exit()
        
    return response

# convert Uniprot accessions to uniprot id or primaryaccession to consume
def get_equivalence_UniprotID(WEBSITE_API, ID):
    r = get_url(f"{WEBSITE_API}/uniprotkb/{ID}")
    output = r.json()['primaryAccession']
    return output

import profet
from profet import Fetcher
from profet import alphafold
from profet import pdb
import requests, sys, json

def download_UniprotID_Alphafold_Structures(df, tmpdir, label_dir, label_name):
        
    #ONLY_ALPHAFOLD = "F4HvG8"
    #ONLY_PDB = "7U6Q"
    import io
    from contextlib import redirect_stdout
    
    WEBSITE_API = 'https://rest.uniprot.org/'
    PROTEINS_API = 'https://www.ebi.ac.uk/proteins/api'
    
    fetcher = Fetcher()
    fetcher.set_directory(str(tmpdir))
    fetcher.set_default_db("alphafold")
    Uniprot_ID_list =df.protein_id.tolist()
    

    open(label_dir+label_name+'.txt', "a").write("Original_ID\tUniprot_ID\tStructure_Alphafold\tSource\n")
    
    for data in tqdm(Uniprot_ID_list):
        Uniprot_ID = get_equivalence_UniprotID(WEBSITE_API, data)
        f = io.StringIO()

        try:
            with redirect_stdout(f):
                filename, _ = fetcher.get_file(Uniprot_ID, filesave=True, db='alphafold')
                out = f.getvalue().split('\n')[-2].split(':')[1].strip()
                open(label_dir+label_name+'.txt', "a").write(data+'\t'+Uniprot_ID+'\t'+filename+'\t'+out+'\n')
        except:
            open(label_dir+label_name+'.txt', "a").write(data+'\t'+Uniprot_ID+'\t'+'No structure available\tNA'+'\n')

'''----------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------'''
'''                               Methods from Gustav to agilize reproducibility of his results              '''
'''----------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------'''



def handle_nans(df, target_columns, method='remove', ):
    '''Function to remove NAs. Either by removing the whole row or by taking the mean of the other values'''
    if method == 'remove':
        obs_before_drop = len(df)
        df = df.dropna(subset=target_columns)
        obs_after_drop = len(df)
        print(obs_before_drop - obs_after_drop, 'observations were removed which had one or more unknown Tm')
        print('Final number of proteins:', obs_after_drop)
    
    elif method == 'mean':
        num_na = df.isna().sum().sum()
        df = df.T.fillna(df.mean(axis=1)).T
        obs_before_drop = len(df)
        df = df.dropna()
        obs_after_drop = len(df)
        print('Filled', num_na, 'Nan cells with mean values (Check that there are only numerical observations in the pH columns)')
        print(obs_before_drop - obs_after_drop, 'observations were removed which could not be interpolated')
        
    elif method == 'keep':
        print('Keeping all nan value')
        
    return df


from Bio import SeqIO
def load_fasta_to_df(filename):
    with open(filename) as fasta_file:  # Will close handle cleanly
        identifiers = []
        seqs = []
        for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
            identifiers.append(seq_record.id)
            seqs.append(str(seq_record.seq))
    #Gathering Series into a pandas DataFrame and rename index as ID column
    df = pd.DataFrame(dict(key=identifiers, sequence=seqs)).set_index(['key'])
    
    return df

'''----------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------------------------'''
import esm
def get_guided_encoder_output(model, alphabet, coords, seq):
    
    device = next(model.parameters()).device
    batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
    batch = [(coords, None, seq)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)
    encoder_out = model.encoder.forward(coords, padding_mask, confidence,
            return_all_hiddens=False)
    # remove beginning and end (bos and eos tokens)
    return encoder_out['encoder_out'][0][1:-1, 0]


def ESM2_IF_repr(df, model, alphabet, folder_path, pooling=True):
    embeddings = torch.zeros(len(df),512)

    with torch.no_grad():
        for i in tqdm(range(0,len(df))):
            #fpath = train.iloc[i].Structure_Alphafold THIS IS A SUPERSTUPID BUG
            fpath = df.iloc[i].Structure_Alphafold
            structure = esm.inverse_folding.util.load_structure(fpath)
            coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)

            rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
            #guide_rep = get_guided_encoder_output(model, alphabet, coords, native_seq)
            torch.save(rep.cpu().detach(), f"{folder_path}/{str(i)+'_'+df.iloc[i].protein_id}.pt")
            
            if pooling==True:
                rep = rep.mean(0).reshape(1,-1)
                embeddings[i]=rep
            
            #torch.save(rep.cpu().detach(), f"{output_dir}/{path[:(len(path)-trim_length)]}.pt")
            
    return embeddings