import torch
import numpy as np
import math, os
from .utilities import plot_results
from torchmetrics import SpearmanCorrCoef
import ipdb
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

class Trainer_Seq_Struct():
    def __init__(self):
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
        
        
        
    def train_LLMRegresor(self,train_iterator, val_iterator, model, device, criterion, optimizer, epoch_num, path_progres, checkpoint_path):
        
        train_loss = []
        val_loss = []
        best_fit = 1e6; best_epoch = 0

        #ipdb.set_trace()        
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

                optimizer.zero_grad()
                
                data = batch_embedding(input).to(dtype=torch.float32, device=device)
                label = torch.tensor(label).to(dtype=torch.float32, device=device) 
                out = model(data)
                loss = criterion(out, label.unsqueeze(-1))
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
                    lossval = criterion(outval, labelt.unsqueeze(-1)) # Calculate validation loss 
                    val_loss_epochs.append(lossval.item())

                #ipdb.set_trace()
                val_loss_epochs = np.mean(val_loss_epochs)
                val_loss.append(round(val_loss_epochs, 3))
                
                train_loss_epochs = np.mean(train_loss_epochs)
                train_loss.append( round( train_loss_epochs ,3) )

            #ipdb.set_trace()
            print('epoch: %d loss: %.3f val loss: %.3f' % (epoch + 1, train_loss_epochs, val_loss_epochs))
            if val_loss_epochs < best_fit:
                best_fit = val_loss_epochs
                best_epoch = epoch
                self.save_checkpoint( best_epoch, model, round(best_fit,3) , 
                                     checkpoint_path+'epoch_{}_loss_val_{}.pt'.format(best_epoch,round(best_fit,3)), 
                                     optimizer)
            
            
            with open(checkpoint_path+path_progres, "a") as myfile:
                myfile.write( 'epoch: %d loss: %.3f val loss: %.3f \n' % (epoch + 1, np.mean(train_loss_epochs), round(val_loss_epochs,3)) )
                
            #progress_bar.update( epoch)
            #progress_bar.set_postfix_str( 'epoch: %d loss: %.3f val loss: %.3f' % (epoch + 1, np.mean(train_loss_epochs), val_loss_epoch) )
            
        
        import matplotlib.pyplot as plt
        
        epochs_domain = range(0, len(train_loss))
        plt.plot(epochs_domain, train_loss, 'b', label='Train Loss')
        plt.plot(epochs_domain, val_loss, 'r', label='Val Loss')
        plt.title('Train ad Evaluation Losss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Evolution')
        plt.legend(), plt.show()
        
        #ipdb.set_trace()
        model, _, epoch_desc, vloss = \
                    self.load_checkpoint(
                        checkpoint_path+'epoch_{}_loss_val_{}.pt'.format(best_epoch,round(best_fit,3)), model, optimizer
                        )
                    
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
        list_outcomes=[]; list_test_labels=[]
        
        batch_test_ensembled = test_loader.dataset.getdata_fromQuery
        
        with torch.no_grad():
            model.eval()
            for test_setidx, labels in tqdm(test_loader, desc ='testing data: ', leave=False):

                test_set = batch_test_ensembled(test_setidx).to(dtype=torch.float32, device=device)
                test_labels = torch.tensor(labels).to(dtype=torch.float32, device=device) 
                outcome = model(test_set)
                list_outcomes.append(outcome.flatten())
                list_test_labels.append(test_labels.flatten())
                
        #ipdb.set_trace()
        pred = torch.hstack(list_outcomes).unsqueeze(-1)
        true_labels = torch.hstack(list_test_labels).unsqueeze(-1)
        
        #loss_test = loss(outcome, test_labels.unsqueeze(-1))
        loss_test = loss(pred, true_labels)
        mae = torch.nn.L1Loss()(pred, true_labels)
        
        spearman = SpearmanCorrCoef()
        spear_corr = spearman(pred.flatten(), true_labels.flatten())
        print('MSE: ' + str(loss_test))
        print('RMSE: ' + str(torch.sqrt(loss_test)))
        print('MAE: ' + str(mae))
        print('Spearman Corr: ' + str(spear_corr))
        
        return true_labels, pred
