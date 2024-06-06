
'''--------------------------------------------------------------------------------------------'''
'''   RANK-N CONTRAST LOSS TAKEN FROM https://github.com/kaiwenzha/Rank-N-Contrast/tree/main'''
'''--------------------------------------------------------------------------------------------'''


import torch
import torch.nn as nn
import torch.nn.functional as F



class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        elif self.distance_type =='l2':
            return (labels[:, None, :] - labels[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)



class BiasOrganismLoss(torch.nn.Module):
    
    def __init__(self):
        super(BiasOrganismLoss, self).__init__()
        #self.device = device
        self.MSE = torch.nn.MSELoss()        
             
    def forward(self, predOGTTM: torch.tensor , target: torch.tensor, avg_TmOrganism: torch.tensor):
        #import ipdb; ipdb.set_trace()
        #return self.MSE( (predOGTTM[:,0] ).unsqueeze(-1), target.to(self.device))
        return self.MSE( (predOGTTM[:,0] + predOGTTM[:,1]).unsqueeze(-1), target)  + self.MSE( predOGTTM[:,0].unsqueeze(-1) , avg_TmOrganism) 


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]
        #import ipdb; ipdb.set_trace()
        #import ipdb; ipdb.set_trace()
        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss
    


loss_dict = {
    'rank_n_contrast_loss_representation': [RnCLoss, 'representation'],
    'rank_n_contrast_loss_predictor': [torch.nn.L1Loss,'L1'],
    'rank_n_contrast_loss_predictorl2': [torch.nn.MSELoss,'L2'],
    'bias_organism_loss_batches': [BiasOrganismLoss, 'batches'],
    'bias_organism_loss_OGT': [BiasOrganismLoss, 'globalOGT'],
    'MSE_loss': [torch.nn.MSELoss, None]
}

class loss_manager(nn.Module):
    def __init__(self, **kwargs ):#lossdevice='cuda'):
        #import ipdb; ipdb.set_trace()
        super(loss_manager, self).__init__()
        self.loss_selector=kwargs['loss']
        self.device=kwargs['device']
        #self.df_training_info=kwargs['df_training_info']
        self.to(self.device)
        self.loss_name, self.misc = loss_dict[self.loss_selector]
        self.loss_base = self.loss_name().to(self.device)

    def eval_model(self,model, data, df, idx, label):
        loss_classname = self.loss_base.__class__.__name__
        
        #import ipdb; ipdb.set_trace()
        if self.loss_base.__class__.__name__ == 'RnCLoss':
            #data_aug = self._data_augmentation_transformation( data, df, idx, label)
            data_aug = self._data_augmentation_transformation2( data, df, idx, label)

            return model(data_aug)
        else:
            return model(data)
        
    def update_loss_setup(self, loss, device):
        self.loss_selector = loss
        self.loss_name, self.misc = loss_dict[self.loss_selector]
        self.loss_base = self.loss_name().to(self.device)
        self.to(self.device)
    
    def _data_augmentation_transformation(self, data, df, idx, label):
        import numpy as np
        
        df_copy=df.loc[idx].copy()
        df_copy['tmp_idx']=np.arange(0,df_copy.shape[0])
        data_clone = data.clone()
        
        import ipdb; ipdb.set_trace()
        for group_name, group_organism in df_copy.groupby('organism'):
            #print(f"Category: {group_name}")
            organism_idx = group_organism.tmp_idx.tolist()
            data_clone[organism_idx] = self._gaussian_noise_contamination(data_clone[organism_idx,:,:], std_portion=0.05)
            
            #print("-------------------")
        
        return torch.vstack([data,data_clone])
    
    def _data_augmentation_transformation2(self, data, df, idx, label):
        import numpy as np
        from torch.distributions import Normal
        
        df_copy=df.loc[idx].copy()
        df_copy['tmp_idx']=np.arange(0,df_copy.shape[0])
        data_clone = torch.zeros(data.shape)
        
                
        #import ipdb; ipdb.set_trace()
        for ii in range(data.shape[0]):
            #print(f"Category: {group_name}")
            # Best one:    
            # sample = Normal( data[ii], torch.ones(data[ii].shape, device='cuda')*0.00001 ).rsample()
            # data_clone[ii] = sample
            
            # Get all the elements that are not zero in a way of mask
            non_zero_mask = (data[ii] != 0).all(dim=1)
            # Get the non-padding embedded protein
            data_without_padding=data[ii][non_zero_mask].reshape(-1,data[ii].shape[-1])
            # Get the batch size of non-padding embedded protein
            data_wt_pd_shape = data_without_padding.shape
            # Sample a Normal distributed embedding, with mean equal to embedding and
            # very narrow variance.
            sample = Normal( data_without_padding, 
                            torch.ones(data_without_padding.shape, 
                                       device='cuda')*0.01 ).rsample()
            # save the sampled augmented protein, preserving the padding with zero value.
            data_clone[ii,0:data_wt_pd_shape[0],:] = sample

        
        return torch.vstack([data,data_clone.to(self.device)])
    
    
    def _gaussian_noise_contamination(self, batch_samples, std_portion=0.0001):
        
        return batch_samples + torch.normal( torch.zeros_like(batch_samples) , 
                                            batch_samples.std(dim=0, keepdim=True) )*std_portion
        '''
        batch_samples + torch.normal( torch.zeros_like(batch_samples.mean(dim=0)) , 
                                                batch_samples.std(dim=0) )*std_portion
        '''
    
    def _batch_conversion_to_RankNContrast_way(self, batch_aug_samples):
        #tmp=torch.vstack([aa,bb])
        #torch.split(tmp,[3,3],dim=0), split normal tensor vs augmented contaminated 
        # tensor per initial batch size chunks along the dimension to split which are the
        # samples
        #return torch.stack([batch_samples,t_batch_samples],dim=1)
        batch_aug_size = batch_aug_samples.shape[0]
        batch_samples, batch_t_samples =  torch.split(batch_aug_samples,
                                                      [batch_aug_size//2 , batch_aug_size//2],
                                                      dim=0)
        return torch.stack([batch_samples,batch_t_samples],dim=1)
    
    def loss_evaluation(self, output, labels, df_training_info, sample_idx):
        
        loss_classname = self.loss_base.__class__.__name__
        
        ''' Bias Loss throughout average tm per organism'''
        if loss_classname == 'BiasOrganismLoss':
            if self.misc == 'batches':
                
                
                organism_Tm_batches = torch.tensor(
                    df_training_info.loc[sample_idx].tm_organism.to_numpy())\
                                                        .to(dtype=torch.float32, device=self.device) 
                
                
                '''
                tmp_df = df_training_info.loc[sample_idx]
                organism_Tm_batches = torch.tensor( tmp_df.organism.map( 
                                                tmp_df.groupby('organism').apply(
                                                lambda a: a.target.astype(float).mean()) ).to_numpy() )\
                                                    .to(dtype=torch.float32, device=self.device) 
                '''
            else:
                organism_Tm_batches = torch.tensor(
                    df_training_info.loc[sample_idx].OGTest_organism.to_numpy())\
                                                        .to(dtype=torch.float32, device=self.device)
            loss = self.loss_base( output , labels.unsqueeze(-1), organism_Tm_batches.unsqueeze(-1) )
            
            ''' Multistep Rank N Constrastive Loss for representation and prediction blocks'''
        elif self.loss_base.__class__.__name__ == 'RnCLoss':
            
            if self.misc == 'representation':
                #import ipdb; ipdb.set_trace()
                # Input for rank n contrast loss
                # features: [bs, 2, feat_dim]
                # labels: [bs, label_dim]
                output_rankN = self._batch_conversion_to_RankNContrast_way(output)
                loss = self.loss_base( output_rankN, labels.reshape(-1,1) )
            else:
                #loss = self.loss_base( output, labels)
                loss = self.loss_base( output, labels.reshape(-1,1))
        else:
            #import ipdb; ipdb.set_trace()
            #loss = self.loss_base( output, labels)
            loss = self.loss_base( output, labels.reshape(-1,1))
            
        return loss
    
    
    