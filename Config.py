
import numpy as np
#Load dataset

data_simulated = np.load("/Users/mayssasoussia/Downloads/DGN-master/simulated dataset/example.npy")

#Number of training epochs
N_max_epochs = 100

#Apply early stopping True/False
early_stop =  True

#Random subset size for SNL function
random_sample_size = 10

#Number of nodes 
N_Nodes = 35

#Number of cross validation folds
n_folds = 5

#Learning Rate for Adam optimizer
lr = 0.0005

#Name of the model
model_name = "Gen_HNN"

#Input, hidden and output features dimensions
in_ch = 140
n_hid = 128
out_ch = 98


CONFIG = {
        "data_simulated":data_simulated, 
        "in_ch":in_ch,
        "n_hid":n_hid,
        "N_max_epochs": N_max_epochs,
        "n_folds": n_folds,
        "random_sample_size": random_sample_size,
        "early_stop": early_stop,
        "model_name": model_name
    }

MODEL_PARAMS = {
        "N_ROIs": N_Nodes,
        "learning_rate" : lr
    }