import torch
import numpy as np
import math
from ..utilities import plot_results
from torchmetrics import SpearmanCorrCoef

class Trainer():
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
    
    
    def train_LLMRegresor(self,train_iterator, val_iterator, model, 
                          device, criterion, optimizer, epoch_num, path_progres, checkpoint_path):
        
        import os
        train_loss = []
        val_loss = []
        best_fit = 1e6; best_epoch = 0

        #import ipdb; ipdb.set_trace()        
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
            print("Directory '%s' created" %checkpoint_path)
        
        model.to(device)

        for epoch in range(epoch_num):
            model.train() 
            train_loss_epochs = []
            for i, (input, label) in enumerate(train_iterator):
                optimizer.zero_grad()
                
                input = input.to(dtype=torch.float32, device=device)
                label = label.to(dtype=torch.float32, device=device) 
                out = model(input)
                loss = criterion(out, label.unsqueeze(-1))
                train_loss_epochs.append(loss.item())
                loss.backward() 
                
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                
                optimizer.step()

            with torch.no_grad(): # evaluate validation loss here 

                model.eval()
                val_loss_epochs = []

                for (inputval, labelval) in val_iterator:
                    
                    inputval = inputval.to(device)
                    labelval = labelval.to(device)
                    outval = model(inputval)
                    lossval = criterion(outval, labelval.unsqueeze(-1)) # Calculate validation loss 
                    val_loss_epochs.append(lossval.item())
                    
                val_loss_epoch = np.mean(val_loss_epochs)
                val_loss.append(round(val_loss_epoch, 3))
                
                train_loss_epochs = round(np.mean(train_loss_epochs),3)
                train_loss.append(train_loss_epochs)

            print('epoch: %d loss: %.3f val loss: %.3f' % (epoch + 1, train_loss_epochs, val_loss_epoch))
            if val_loss_epoch < best_fit:
                best_fit = val_loss_epoch
                best_epoch = epoch
                self.save_checkpoint( best_epoch, model, round(best_fit,3) , 
                                     checkpoint_path+'epoch_{}_loss_val_{}.pt'.format(best_epoch,round(best_fit,3)), 
                                     optimizer)
            
            
            with open(checkpoint_path+path_progres, "a") as myfile:
                myfile.write( 'epoch: %d loss: %.3f val loss: %.3f \n' % (epoch + 1, train_loss_epochs, round(val_loss_epoch,3)) )
            

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

        #return model, val_loss
    
    def test_model(self, model, test_set, test_labels, loss, device):
        test_set = test_set.to(dtype=torch.float32, device=device)
        test_labels = test_labels.to(dtype=torch.float32, device=device) 
        outcome = model(test_set)

        loss_test = loss(outcome, test_labels.unsqueeze(-1))
        mae = torch.nn.L1Loss()(outcome.flatten(), test_labels)
        
        spearman = SpearmanCorrCoef()
        spear_corr = spearman(outcome.flatten(), test_labels)
        print('MSE: ' + str(loss_test))
        print('RMSE: ' + str(torch.sqrt(loss_test)))
        print('MAE: ' + str(mae))
        print('Spearman Corr: ' + str(spear_corr))
        
        return loss_test, outcome
