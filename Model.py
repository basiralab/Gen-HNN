import torch
import random
import math
import uuid
import os
import torch.nn as nn
import numpy as np
from data_helper import preprocess_data_array, get_features
from Hypergraph_utilities import construct_G, construct_H_with_KNN, construct_hypergraph_tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import time
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def show_image(img, i, score):
    img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
    plt.imshow(img)
    plt.title("fold " + str(i) + " Frobenious distance: " +  "{:.2f}".format(score))
    plt.axis('off')
    plt.show()

def clear_dir(dir_name):
    for file in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file))




class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class Gen_HNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.48):
        super(Gen_HNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        H_CBT = construct_H_with_KNN(x.detach().numpy())

        return H_CBT


    def get_params(self, deep=True):

        return {

            'in_ch': self.in_ch,

            'n_hid': self.n_hid,

        }



    @staticmethod
    def generate_subject_biased_cbts(model, train_data):
        """
            Generates all possible HCBTs for a given training set.
            Args:
                model: trained Gen_HNN model
                train_data: list of data objects
        """
        model.eval()
        cbts = np.zeros((35,35,len(train_data))) #model.model_params["N_ROIs") = 35
        train_data = [d.to(device) for d in train_data]
        for i, data in enumerate(train_data):
            #cbt = model(data)
            fts_mat = torch.tensor(get_features(data), dtype=torch.float32)
            G_norm = torch.tensor(construct_G(data.numpy()), dtype=torch.float32)
            cbt = model(fts_mat, G_norm)
            cbt = torch.tensor(cbt, dtype=torch.float32)

            cbts[:,:,i] = np.array(cbt.cpu().detach())

        return cbts


    @staticmethod
    def generate_cbt_median(model, train_data):
        """
            Generate optimized HCBT for the training set (use post training refinement)
            Args:
                model: trained Gen_HNN model
                train_data: list of data objects
        """
        model.eval()
        cbts = []
        train_data = [d.to(device) for d in train_data]
        for data in train_data:
            fts_mat = torch.tensor(get_features(data), dtype=torch.float32)
            G_norm = torch.tensor(construct_G(data.numpy()), dtype=torch.float32)
            cbt = model(fts_mat, G_norm)
            cbt = torch.tensor(cbt, dtype=torch.float32)


            cbts.append(np.array(cbt.cpu().detach()))
        final_cbt = torch.tensor(np.median(cbts, axis = 0), dtype = torch.float32).to(device)

        return final_cbt



    @staticmethod
    def mean_frobenious_distance(generated_cbt, test_data):
      """
          Calculate the mean Frobenious distance between the CBT and test subjects (all views)
          Args:
              generated_cbt: trained DGN model
              test_data: list of data objects
      """
      frobenius_all = []
      for views in test_data:
        #views = torch.tensor(data, dtype=torch.float)
          for index in range(views.shape[2]):
              diff = torch.abs(views[:,:,index] - generated_cbt)
              diff = diff*diff
              sum_of_all = diff.sum()
              d = torch.sqrt(sum_of_all)
              frobenius_all.append(d)
      return sum(frobenius_all) / len(frobenius_all)

    
    @staticmethod
    def mean_frobenious_distance_views(generated_cbt, test_data):
      """
          Calculate the mean Frobenious distance between the CBT and test subjects (all views)
          Args:
              generated_cbt: trained DGN model
              test_data: list of data objects
      """
      frobenius_all = []
      for views in test_data:
        #views = torch.tensor(data, dtype=torch.float)
          for index in range(views.shape[2]):
              diff = torch.abs(views[:,:,index] - generated_cbt)
              diff = diff*diff
              sum_of_all = diff.sum()
              d = torch.sqrt(sum_of_all)
              frobenius_all.append(d)
      return sum(frobenius_all) / len(frobenius_all)

    @staticmethod
    def train_model(X,in_ch, n_hid, name, model_params, n_max_epochs, early_stop, model_name, random_sample_size=10, n_folds=5):

        """
        Trains a model for each cross validation fold and saves all models along with CBTs to ./output/<model_name>
        Args:
            X (np array): dataset (train+test) with shape [N_Subjects, N_ROIs, N_ROIs]
            hypergraphs_tensor (np array): dataset (train +test) converted to high-order with shape [N_Subjects, N_ROIs, N_ROIs]
            name (string): the name of a given class  (e.g., ASD)
            n_max_epochs (int): number of training epochs (if early_stop == True this is maximum epoch limit)
            early_stop (bool): if set true, model will stop training when overfitting starts.
            model_name (string): name for saving the model
            random_sample_size (int): random subset size for SNL function
            n_folds (int): number of cross validation folds
        Return:
            models: trained models
        """
        models = []
        save_path = "output/" + model_name + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_id = str(uuid.uuid4())
        with open(save_path + "model_params.txt", 'w') as f:
            print(model_params, file=f)

        CBTs = []
        scores = []
        for i in range(n_folds):
            torch.cuda.empty_cache()
            print("********* FOLD {} *********".format(i))
            train_data, test_data, train_indices, test_indices = preprocess_data_array(X, number_of_folds=n_folds, current_fold_id=i)

            test_data = np.array([d.astype(np.float32) for d in test_data])
            test_data = torch.tensor(test_data, dtype=torch.float32).to(device)


            loss_weightes = torch.rand(random_sample_size, requires_grad=True)
            loss_weightes = loss_weightes.to(device)


            train_data = np.array([d.astype(np.float32) for d in train_data])
            train_data = torch.tensor(train_data, dtype=torch.float32).to(device)


            model = Gen_HNN_embedding(in_ch, n_hid)
            model = model.to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.00)

            targets = [torch.tensor(tensor, dtype=torch.float32).to(device) for tensor in train_data]
            test_errors = []
            tick = time.time()


            for epoch in range(n_max_epochs):
                model.train()
                losses = []
                for data in train_data:

                    # Compose Dissimilarity matrix from network outputs
                    fts_mat = torch.tensor(get_features(data), dtype=torch.float32)
                    G_norm = torch.tensor(construct_G(data.numpy()), dtype=torch.float32)
                    cbt = model(fts_mat, G_norm)
                    cbt = torch.tensor(cbt, dtype=torch.float32)


                    sampled_targets = random.sample(targets, random_sample_size)
                    sampled_targets_G = [torch.tensor(construct_G(st.numpy()), dtype=torch.float32) for st in sampled_targets]

                    expanded_cbt = cbt.expand((len(sampled_targets_G),model_params["N_ROIs"],model_params["N_ROIs"]))

                    sampled_targets_Gt = torch.stack(sampled_targets_G)
                    diff = torch.abs(expanded_cbt - sampled_targets_Gt)  # Absolute difference


                    sum_of_all = torch.mul(diff, diff).sum(axis=(1, 2))  # Sum of squares
                    l = torch.sqrt(sum_of_all)  # Square root of the sum
                    losses.append((l * loss_weightes[:random_sample_size]).sum())




                # Backprob
                optimizer.zero_grad()
                loss = torch.mean(torch.stack(losses))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Track the loss
                if epoch % 10 == 0:
                    cbt = Gen_HNN_embedding.generate_cbt_median(model, train_data)

                    #test_data_G = [torch.tensor(construct_G(td.numpy()), dtype=torch.float32) for td in test_data]
                    hypergraphs_tensor = construct_hypergraph_tensor(X)
                    test_data_tensor = [hypergraphs_tensor[i] for i in test_indices]
                    test_data_views = [torch.tensor(td, dtype=torch.float32) for td in test_data_tensor]
            
                    rep_loss = Gen_HNN_embedding.mean_frobenious_distance_views(cbt, test_data_views)
                    tock = time.time()
                    time_elapsed = tock - tick
                    tick = tock
                    rep_loss = float(rep_loss)
                    test_errors.append(rep_loss)
                    print("Epoch: {}  |  Test Rep: {:.2f}  |  Time Elapsed: {:.2f}  |".format(epoch, rep_loss, time_elapsed))
                    # Early stopping control
                    if len(test_errors) > 6 and early_stop:
                        torch.save(model.state_dict(), "temp/weight_" + model_id + "_" + str(rep_loss)[:5] + ".model")
                        last_6 = test_errors[-6:]
                        if (all(last_6[i] < last_6[i + 1] for i in range(5))):
                            print("Early Stopping")
                            break


            ################# SAVE SAVE SAVE ###################
            #save the losses
            np.save(name+ "_fold_" + str(i) + "losses", test_errors)
            ################# SAVE SAVE SAVE ###################


            # Restore best model so far
            try:
                restore = "./temp/weight_" + model_id + "_" + str(min(test_errors))[:5] + ".model"
                model.load_state_dict(torch.load(restore))
            except:
                pass
            #torch.save(model.state_dict(), save_path + "fold" + str(i) + ".model")
            models.append(model)

            # Generate and save refined CBT
            cbt = Gen_HNN_embedding.generate_cbt_median(model, train_data)
            rep_loss = Gen_HNN_embedding.mean_frobenious_distance_views(cbt, test_data_views)
            cbt = cbt.cpu().numpy()
            CBTs.append(cbt)

            ################# SAVE SAVE SAVE ###################
            np.save(save_path + name + "_fold" + str(i) + "_cbt", cbt)

            # Save all subject biased CBTs
            all_cbts = Gen_HNN_embedding.generate_subject_biased_cbts(model, train_data)
            np.save(save_path + name + "_fold" + str(i) + "_all_cbts", all_cbts)

            ################# SAVE SAVE SAVE ###################

            scores.append(float(rep_loss))
            print("FINAL RESULTS  REP: {}".format(rep_loss))

            # Clean interim model weights
            clear_dir("temp")

        for i, cbt in enumerate(CBTs):
            show_image(cbt, i, scores[i])
        avg_scores = np.mean(scores)
        np.save(name + "_avg_scores",avg_scores )

        return models


