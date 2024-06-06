import torch
import numpy as np
import math, os
from .utilities import plot_results
from torchmetrics import SpearmanCorrCoef
import ipdb
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

class Trainer_Seq_Struct2():
    def __init__(self, avgtm_mode='batches'):
        self.avgtm_mode=avgtm_mode
        pass
    
    def save_checkpoint(self, EPOCH, net, LOSS, PATH, optimizer):
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
    
    def load_checkpoint(self,PATH, model, optimizer):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss
        
        
        
    def train_LLMRegresor(self,train_iterator, val_iterator, model, device, criterion, optimizer, epoch_num, path_progres, checkpoint_path, **kargs):
        
        import copy
        train_loss = []
        val_loss = []
        best_fit = 1e6; best_epoch = 0
        best_model = copy.deepcopy(model)

        #import ipdb; ipdb.set_trace()        
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            print("Directory '%s' created" %checkpoint_path)
            
        batch_embedding = train_iterator.dataset.getdata_fromQuery
        batch_val_embedding = val_iterator.dataset.getdata_fromQuery
        
        model.to(device)
        #progress_bar = tqdm(desc='Epochs '  + '/' + str(epoch_num), total=epoch_num )

        for epoch in tqdm( range(epoch_num) ):
            model.train() 
            train_loss_epochs = []
            for input, label in tqdm(train_iterator, desc ='training epoch by batches: ', leave=False):
                #import ipdb; ipdb.set_trace()
                optimizer.zero_grad()
                
                data = batch_embedding(input).to(dtype=torch.float32, device=device)
                label = torch.tensor(label).to(dtype=torch.float32, device=device) 
                out = model(data)
                
                #import ipdb; ipdb.set_trace()
                if criterion.__class__.__name__ == 'BiasOrganismLoss':
                    if self.avgtm_mode=='batches':
                        organism_Tm_batches_train = torch.tensor(train_iterator.dataset.data.loc[input].tm_organism.to_numpy())\
                                                                                                .to(dtype=torch.float32, device=device) 
                    else:
                        organism_Tm_batches_train = torch.tensor(train_iterator.dataset.data.loc[input].OGTest_organism.to_numpy())\
                                                                                                .to(dtype=torch.float32, device=device) 
                        
                    loss = criterion( out , label.unsqueeze(-1), organism_Tm_batches_train.unsqueeze(-1) )
                else:
                    loss = criterion( out , label.unsqueeze(-1) )
                    
                train_loss_epochs.append(loss.item())
                loss.backward() 
                
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                
                optimizer.step()

            #ipdb.set_trace()        
            with torch.no_grad(): # evaluate validation loss here 

                model.eval()
                val_loss_epochs = []
                #print('model in validation set:... ')
                #for (inputval, labelval) in tqdm(val_iterator, desc='model in validation set: ', leave=False):
                for (inputval, labelval) in val_iterator:#, desc='model in validation set: ', leave=False):
                    
                    #inputval = inputval.to(device)
                    #labelval = labelval.to(device)
                    datat = batch_val_embedding(inputval).to(dtype=torch.float32, device=device)
                    labelt = torch.tensor(labelval).to(dtype=torch.float32, device=device) 
                    outval = model(datat)
                
                    if criterion.__class__.__name__ == 'BiasOrganismLoss':
                        if self.avgtm_mode=='batches':
                            organism_Tm_batches_val = torch.tensor(val_iterator.dataset.data.loc[inputval].tm_organism.to_numpy())\
                                                                                                            .to(dtype=torch.float32, device=device)

                        else:
                            organism_Tm_batches_val = torch.tensor(val_iterator.dataset.data.loc[inputval].OGTest_organism.to_numpy())\
                                                                                                .to(dtype=torch.float32, device=device) 
                            
                        lossval = criterion( outval , labelt.unsqueeze(-1), organism_Tm_batches_val.unsqueeze(-1) )
                    else:
                        lossval = criterion( outval , labelt.unsqueeze(-1) )
                    

                    val_loss_epochs.append(lossval.item())

                #ipdb.set_trace()
                val_loss_epochs = np.mean(val_loss_epochs)
                val_loss.append(round(val_loss_epochs, 3))
                
                train_loss_epochs = np.mean(train_loss_epochs)
                train_loss.append( round( train_loss_epochs ,3) )

            #import ipdb; ipdb.set_trace()
            print('epoch: %d loss: %.3f val loss: %.3f' % (epoch + 1, train_loss_epochs, val_loss_epochs))
            if val_loss_epochs < best_fit:
                best_fit = val_loss_epochs
                best_model = copy.deepcopy(model)
                best_epoch = epoch
            
            """
            if epoch % 10 == 0:
                import ipdb; ipdb.set_trace()
                self.save_checkpoint( best_epoch, best_model, round(best_fit,3) , 
                                    checkpoint_path+'epoch_{}_loss_val_{}.pt'.format(best_epoch,round(best_fit,3)), 
                                    optimizer)
            """
            
            with open(checkpoint_path+path_progres, "a") as myfile:
                myfile.write( 'epoch: %d loss: %.3f val loss: %.3f \n' % (epoch + 1, np.mean(train_loss_epochs), round(val_loss_epochs,3)) )
                myfile.close()
                

        
        #ipdb.set_trace()
        """
        model, _, epoch_desc, vloss = \
                    self.load_checkpoint(
                        checkpoint_path+'epoch_{}_loss_val_{}.pt'.format(best_epoch,round(best_fit,3)), model, optimizer
                        )
        """
        self.save_checkpoint( best_epoch, best_model, round(best_fit,3) , 
                                    checkpoint_path+'epoch_{}_loss_val_{}.pt'.format(best_epoch,round(best_fit,3)), 
                                    optimizer)
        
        model, vloss = best_model, round(best_fit,3)
                    
        return model, vloss
    
    
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
        
    
    def test_model(self, model, test_loader, loss, device):
        #ipdb.set_trace()
        list_outcomes=[]; list_test_labels=[]; tmp_OGTerrorEcoli=[]
        
        batch_test_ensembled = test_loader.dataset.getdata_fromQuery
        
        with torch.no_grad():
            model.eval()
            for test_setidx, labels in tqdm(test_loader, desc ='testing data: ', leave=False):

                test_set = batch_test_ensembled(test_setidx).to(dtype=torch.float32, device=device)
                test_labels = torch.tensor(labels).to(dtype=torch.float32, device=device) 
                outcome = model(test_set)
                if outcome.shape[1]==2:
                    list_outcomes.append( (outcome[:,0]+outcome[:,1]).flatten() )
                else:
                    list_outcomes.append(outcome.flatten())
                list_test_labels.append(test_labels.flatten())
                #tmp_OGTerrorEcoli.append(outcome[:,0]-50.841375)
                
                
        #ipdb.set_trace()
        pred = torch.hstack(list_outcomes).unsqueeze(-1)
        true_labels = torch.hstack(list_test_labels).unsqueeze(-1)
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
        #self.plot_overlaped_histograms(pred.detach().cpu().flatten().numpy(), true_labels.detach().cpu().flatten().numpy())
        #self.plot_histogram( (pred - true_labels).detach().cpu().flatten().numpy(),'Tm error', 'prediction error distribution')
        #self.plot_histogram( ecoli_ogt_error.detach().cpu().flatten().numpy(),'OGT ecoli error', 'OGT error distribution')
        
        return true_labels, pred
