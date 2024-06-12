import torch
import numpy as np
import math, os
from .utilities import plot_results
from torchmetrics import SpearmanCorrCoef
import ipdb
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class Trainer_Seq_Struct_general():
    def __init__(self, 
                 model,
                 yaml_pretrained_info,
                 yaml_path_progress_info,
                 yaml_output_info,
                 train_iterator, 
                 val_iterator,
                 loss_manager,
                 device):
        self.yaml_pretrained_info = yaml_pretrained_info
        self.yaml_path_progress_info = yaml_path_progress_info
        self.yaml_output_info = yaml_output_info
        self.model = model
        self.device= device
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.loss_manager = loss_manager

    
    def save_checkpoint(self, EPOCH, net, LOSS, PATH):
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'loss': LOSS,
            }, PATH)
    
    def load_checkpoint(self,PATH, model):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, epoch, loss
        
    def search_pretrained(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            print("Directory '%s' created" %checkpoint_path)
        return checkpoint_path

    
    def __check_suffix_in_files(self, main_path, suffix):
        # check the existence of the repr and pred folders
        first_val =  any( self.yaml_pretrained_info[ suffix ] in ii 
                   for ii in os.listdir(self.yaml_pretrained_info[main_path]) )
        # check if that folder is empty or not (second_val=False in case of being empty)
        if not os.listdir(self.yaml_pretrained_info[main_path] + self.yaml_pretrained_info[suffix]):
            second_val = False
        else: second_val = True
        return first_val and second_val
    
    def filter_pt_files(self, root_path):
        paths = os.listdir( root_path )
        pt_files = [path for path in paths if path.endswith('.pt')]
        return pt_files[0]

    def __get_checkpoint_from_yaml(self, model, tag_main_path, tag_model, device):
        
        root_path = f"{self.yaml_pretrained_info[tag_main_path]}{self.yaml_pretrained_info[tag_model]}/"
        best_checkpoint = self.filter_pt_files(root_path)#os.listdir( root_path )[-1]
        
        model_ck, _, _ =\
                    self.load_checkpoint(f"{root_path}{best_checkpoint}", model)

        return model_ck.to(self.device)
    
    def status_model_weights(self, model):
        # Check which layers are frozen
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
    
    def setup_training(self):
        
        import ipdb; ipdb.set_trace()
        # paths to search pretrained models

        path_pred = self.search_pretrained( self.yaml_pretrained_info['main_path']  + self.yaml_pretrained_info['pred_suffixe'])

        # if representation suffixe exist on yaml file, this will correspond to rank n contrastive loss, 
        # otherwise the normal loss evaluation will be evaluated accordingly 
        if 'representation_suffixe' in self.yaml_pretrained_info:
            
            path_repr = self.search_pretrained( self.yaml_pretrained_info['main_path']  + self.yaml_pretrained_info['representation_suffixe'])
            
            # check the existence of a pretrained model for 
            # representation tunned with rank-n-constrative loss

            if self.__check_suffix_in_files('main_path', 'representation_suffixe'):
                
                self.model.encoder.LAmodule = self.__get_checkpoint_from_yaml(self.model.encoder.LAmodule, 
                                'main_path', 'representation_suffixe', self.device)
                
                #import ipdb; ipdb.set_trace()
                # check the existence of a pretrained model for predictor as second scheme on rank-n-constrative 
                # framework, otherwise the predictor would be trained
                if self.__check_suffix_in_files('main_path', 'pred_suffixe'):
                    self.model.encoder.FFNN = self.__get_checkpoint_from_yaml(self.model.encoder.FFNN, 
                                'main_path', 'pred_suffixe', self.device)
                
                else:
                    print('Starting regresor training using the pretrained rank-n-contrastive representation....\n\n')
                    self.model.setup_tunning(mode='predictor')
                    self.search_pretrained(path_pred)
                    
                    opt = self.model.set_optimizer()
                    self.loss_manager.update_loss_setup('rank_n_contrast_loss_predictorl2', self.device)
                    
                    self.model.encoder = self.train_model( self.model.encoder, 
                                                       opt['predictor_opt'], epoch_num=200, folder_path=path_pred)
                    
                    #import ipdb; ipdb.set_trace()
                    print('Done')
                    
            # Otherwise train the representation first throughout 
            # Rank-n_contrastive loss, then the predictor with l1 los
            #         print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")
            else:
                print('Training representation using Rank N Contrastive Loss - Light Attention Block...\n\n')
                self.model.setup_tunning(mode='representation')
                self.search_pretrained(path_repr)
                opt = self.model.set_optimizer()
                self.model.encoder.LAmodule = self.train_model(self.model.encoder.LAmodule, 
                                                    opt['Representation_opt'], epoch_num=200, folder_path=path_repr)
                
                #import ipdb; ipdb.set_trace()
                print('Starting regresor training using the pretrained rank-n-contrastive representation....\n\n')
                self.model.setup_tunning(mode='predictor')
                self.search_pretrained(path_pred)
                self.loss_manager.update_loss_setup('rank_n_contrast_loss_predictorl2', self.device)

                #self.search_pretrained( self.yaml_pretrained_info['main_path'] + '/' + self.yaml_pretrained_info['pred_suffixe'])
                self.model.encoder = self.train_model( self.model.encoder, 
                                    opt['predictor_opt'], epoch_num=200, folder_path=path_pred)
        
        # if representation suffixe exist on yaml file, this will correspond to rank n contrastive loss, 
        # otherwise the normal loss evaluation will be evaluated accordingly
        else:
            #import ipdb; ipdb.set_trace()
            if self.__check_suffix_in_files('main_path', 'pred_suffixe'):
                print('Loading global model....\n\n')
                self.model.encoder = self.__get_checkpoint_from_yaml(self.model.encoder, 
                                'main_path', 'pred_suffixe', self.device)
            else:
                print('Training whole model using {} loss....\n\n'.format(self.loss_manager.loss_name.__name__))
                self.model.setup_tunning(mode='both')
                self.search_pretrained(path_pred)
                opt = self.model.set_optimizer()
                self.model.encoder = self.train_model(self.model.encoder, 
                                    opt['predictor_opt'], epoch_num=200, folder_path=path_pred)
            
        return self.model.encoder
        
    
    def __train_epoch(self, model, data, label,  idx, df, historic_epochs):        
        out = self.loss_manager.eval_model(model, data, df, idx, label)
        loss = self.loss_manager.loss_evaluation(out, label, df, idx)
        historic_epochs.append(loss.item())
        return loss
        

    def train_model(self, model, optimizer, epoch_num=200, **kwargs):    
    
        import copy
        train_loss = []
        val_loss = []
        best_fit = 1e6; best_epoch = 0
        checkpoint_path = kwargs['folder_path']
        best_model = copy.deepcopy(model)

            
        batch_embedding = self.train_iterator.dataset.getdata_fromQuery
        batch_val_embedding = self.val_iterator.dataset.getdata_fromQuery
        #import ipdb; ipdb.set_trace()
                
        for epoch in tqdm( range(epoch_num) ):
            model.train() 
            train_loss_epochs = []
            #import ipdb; ipdb.set_trace()
            for input, label in tqdm(self.train_iterator, desc ='training epoch by batches: ', leave=False):
                #import ipdb; ipdb.set_trace()
                optimizer.zero_grad()
                
                data = batch_embedding(input).to(dtype=torch.float32, device=self.device)
                label = torch.tensor(label).to(dtype=torch.float32, device=self.device)                 
                loss_train = self.__train_epoch(model, data, label, input, self.train_iterator.dataset.data, train_loss_epochs)
                loss_train.backward() 
                optimizer.step()

            #ipdb.set_trace()        
            with torch.no_grad(): # evaluate validation loss here 

                model.eval()
                val_loss_epochs = []

                #import ipdb; ipdb.set_trace()
                for (inputval, labelval) in self.val_iterator:
                    
                    datat = batch_val_embedding(inputval).to(dtype=torch.float32, device=self.device)
                    labelt = torch.tensor(labelval).to(dtype=torch.float32, device=self.device)                     
                    loss_val = self.__train_epoch(model, datat, labelt, inputval, self.val_iterator.dataset.data, val_loss_epochs)
                    val_loss_epochs.append(loss_val.item())

                #ipdb.set_trace()
                
                train_loss_epochs = np.mean(train_loss_epochs)
                train_loss.append( round( train_loss_epochs ,3) )
                val_loss_epochs = np.mean(val_loss_epochs)
                val_loss.append(round(val_loss_epochs, 3))
                
            #ipdb.set_trace()
            print('epoch: %d loss: %.3f val loss: %.3f' % (epoch + 1, train_loss_epochs, val_loss_epochs))
            if val_loss_epochs < best_fit:
                best_fit = val_loss_epochs
                best_model = copy.deepcopy(model)
                best_epoch = epoch
            
            """
            print('epoch: %d loss: %.3f val loss: %.3f' % (epoch + 1, train_loss_epochs, val_loss_epochs))
            if val_loss_epochs < best_fit:
                best_fit = val_loss_epochs
                best_epoch = epoch
                self.save_checkpoint( best_epoch, model, round(best_fit,3) , 
                                     checkpoint_path+'epoch_{}_loss_val_{}.pt'.format(best_epoch,round(best_fit,3)), 
                                     optimizer)
            """
            with open(checkpoint_path +'/training_progress.txt', "a") as myfile:
                myfile.write( 'epoch: %d loss: %.3f val loss: %.3f \n' % (epoch + 1, np.mean(train_loss_epochs), round(val_loss_epochs,3)) )
                myfile.close()
                

        
        #ipdb.set_trace()
        """
        model, _, epoch_desc, vloss = \
                    self.load_checkpoint(
                        checkpoint_path+'/epoch_{}_loss_val_{}.pt'.format(best_epoch,round(best_fit,3)), model, optimizer
                        )
        """
        #import ipdb; ipdb.set_trace()
        #self.save_checkpoint( best_epoch, best_model, round(best_fit,3) , 
        #                            checkpoint_path+'/epoch_{}_loss_val_{}.pt'.format(best_epoch,round(best_fit,3)))
        
        self._save_best_model_setup( best_epoch, best_model, round(best_fit,3) , 
                                    checkpoint_path+'/epoch_{}_loss_val_{}.pt'.format(best_epoch,round(best_fit,3)) )
        
        model, vloss = best_model, round(best_fit,3)
                 
        return model
    
    def _save_best_model_setup(self,*args):
        
        # Due to the fact that there are some algorithms in the batch that are 
        # trained sequentially in blocks like the case of Rank N Constrast, it 
        # is necessary to save separatelly (Encoder/predictor=> LAmodule/FFNN), 
        # or complete the rest of cases, it was necessary to ask for the case 
        # in which the models are trained by specifical blocks or not, to save 
        # such models accordingly  
        best_epoch, best_model, best_fit, path = args
        
        if best_model.__class__.__name__ == 'LA_MLP_pred':
            # case when both encoder and regresor layers were trained at the same time
            if self.model.model_mode =='both':
                self.save_checkpoint(*args)
            else:
                # case when the encoder layer was freezed and only the regressor was trained
                tmp_best_model = best_model.FFNN
                self.save_checkpoint(best_epoch, tmp_best_model, best_fit, path)
        else:
            tmp_best_model = best_model
            self.save_checkpoint(best_epoch, tmp_best_model, best_fit, path)
            
                
        
    
    def plot_histogram(self, data, label, title):
        
        plt.figure(figsize =(8,6) )
        sns.histplot(data, bins=30, kde=True, color='gray', label=label )
        plt.xlabel('Tms')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.show()
        
    def plot_overlaped_histograms(self, real, trueval):
        
        plt.figure(figsize =(8,6) )
        sns.histplot(real, bins=30, kde=True, color='blue', label='predicted' )
        sns.histplot(trueval, bins=30, kde=True, color='orange', label='real' )

        plt.xlabel('Tms')
        plt.ylabel('Frequency')
        plt.title('Overlapping Histograms')
        plt.legend()
        plt.show()
        
    
    @staticmethod
    def test_model(model, test_loader, device, output_metrics):

        list_outcomes=[]; list_test_labels=[]; tmp_OGTerrorEcoli=[]
        
        batch_test_ensembled = test_loader.dataset.getdata_fromQuery
        
        #import ipdb; ipdb.set_trace()
        with torch.no_grad():
            model.eval()
            for test_setidx, labels in tqdm(test_loader, desc ='testing data: ', leave=False):

                test_set = batch_test_ensembled(test_setidx).to(dtype=torch.float32, device=device)
                test_labels = torch.tensor(labels).to(dtype=torch.float32, device=device) 
                outcome = model(test_set)
                
                if outcome.shape[1]>1:
                    list_outcomes.append( (outcome[:,0]+outcome[:,1]).flatten() )
                else:
                    list_outcomes.append( outcome[:,0].flatten() )
                    
                list_test_labels.append(test_labels.flatten())
                #tmp_OGTerrorEcoli.append(outcome[:,0]-50.841375)
                
                
        #ipdb.set_trace()
        pred = torch.hstack(list_outcomes).unsqueeze(-1)
        true_labels = torch.hstack(list_test_labels).unsqueeze(-1)
        
        df_predictions = pd.DataFrame({
            'pred': pred.flatten().cpu().numpy(),  # Flatten to make it a 1D array
            'true': true_labels.flatten().cpu().numpy()   # Flatten to make it a 1D array
        })
        #ecoli_ogt_error = torch.hstack(tmp_OGTerrorEcoli).unsqueeze(-1)
        
        #loss_test = loss(outcome, test_labels.unsqueeze(-1))
        loss_test = torch.nn.MSELoss()(pred, true_labels)
        mae = torch.nn.L1Loss()(pred, true_labels)
        
        spearman = SpearmanCorrCoef()
        spear_corr = spearman(pred.flatten(), true_labels.flatten())
        print('MSE: ' + str(loss_test))
        print('RMSE: ' + str(torch.sqrt(loss_test)))
        print('MAE: ' + str(mae))
        print('Spearman Corr: ' + str(spear_corr))
        
        #import ipdb; ipdb.set_trace()
        if not os.path.exists( '/'.join(output_metrics.split('/')[:-1]) ):
            os.makedirs( '/'.join(output_metrics.split('/')[:-1]) )
            
        folder_path = '/'.join(output_metrics.split('/')[:-1])
        root_out_filename = output_metrics.split('/')[-1].split('.')[0]
        
        df_predictions.to_csv( f'{folder_path}/{root_out_filename}_preds.txt', sep='\t', index=False)
        Trainer_Seq_Struct_general.make_scatterplots_pred(pred.detach().cpu().numpy().flatten(), 
                                                          true_labels.detach().cpu().numpy().flatten(), 
                                                          f'{folder_path}/{root_out_filename}.pdf')
        
        with open( output_metrics, "w") as file1:
            file1.write('MSE: ' + str(loss_test) +"\n")
            file1.write('RMSE: ' + str(torch.sqrt(loss_test)) +"\n")
            file1.write('MAE: ' + str(mae) +"\n")
            file1.write('Spearman Corr: ' + str(spear_corr))
        
        #import ipdb; ipdb.set_trace()
        #self.plot_overlaped_histograms(pred.detach().cpu().flatten().numpy(), true_labels.detach().cpu().flatten().numpy())
        #self.plot_histogram( (pred - true_labels).detach().cpu().flatten().numpy(),'Tm error', 'prediction error distribution')
        #self.plot_histogram( ecoli_ogt_error.detach().cpu().flatten().numpy(),'OGT ecoli error', 'OGT error distribution')
        
        return true_labels, pred

    @staticmethod
    def make_scatterplots_pred(pred, target, output):
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from scipy.stats import linregress
        # Generate sample data
        np.random.seed(0)
        x = pred
        y = target
        spearman = SpearmanCorrCoef()
        spear_corr = spearman(torch.from_numpy(pred), torch.from_numpy(target)).item()
        #import ipdb; ipdb.set_trace()
        # Fit a linear regression line
        if np.all(x == x[0]):
            plot_linreg = False
        else:
            plot_linreg = True
            slope, intercept, _, _, _ = linregress(x, y)
            fit_line_x = x
            fit_line_y = slope * fit_line_x + intercept
                    
        # Set style
        sns.set_context("paper")
        sns.set_style("ticks")
        # Create scatterplot
        plt.figure(figsize=(6, 4))
        #sns.scatterplot(x=x, y=y, color="blue", edgecolor="black", alpha=0.5, s=40)
        sns.scatterplot(x=x, y=y, color="blue", edgecolor="black", alpha=0.5, s=10)

        
        # Add regression line
        if plot_linreg == True:
            plt.plot(fit_line_x, fit_line_y, linestyle="--", color="red", linewidth=2)

        # Extend dashed line symmetrically
        #plt.axhline(y=intercept, color="red", linestyle="--", linewidth=2)
        #plt.axvline(x=-intercept/slope, color="red", linestyle="--", linewidth=2)

        # Set labels and title
        plt.xlabel("Predictions", fontsize=12)
        plt.ylabel("Target", fontsize=12)
        plt.title(f"{output.split('/')[-1].split('.')[0]}: Spearman={round(spear_corr,4)}", fontsize=11)
        # Add grid
        plt.grid(True, linestyle="--", alpha=0.5)
        # Add legend
        plt.legend(["Data", "Trend Line"], loc="upper left", fontsize=10)
        # Adjust layout
        plt.tight_layout()
        # Save or show plot
        plt.savefig(output, dpi=300)  # Save plot as PNG with high resolution
        plt.show()
        
        