import os
from Model import Gen_HNN_embedding
from Config import MODEL_PARAMS, CONFIG



if not os.path.exists('temp'):
    os.makedirs('temp')
if not os.path.exists('output'):
    os.makedirs('output')



Gen_HNN_embedding.train_model(
                CONFIG["data_simulated"],
                CONFIG["in_ch"],
                CONFIG["n_hid"],
                name = "ASD_LH",
                model_params=MODEL_PARAMS,
                n_max_epochs=CONFIG["N_max_epochs"],
                n_folds=CONFIG["n_folds"],
                random_sample_size=CONFIG["random_sample_size"],
                early_stop=CONFIG["early_stop"],
                model_name=CONFIG["model_name"])