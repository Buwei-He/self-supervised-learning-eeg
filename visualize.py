
from models.model_factory import Model_factory
from Dataset.dataloader import data_loader, dataset_class
from torch.utils.data import DataLoader
from Dataset.dataloader import dataset_class
from utils.utils import load_model
from models.Series2Vec.S2V_training import S2V_make_representation


def load_config_from_json(root_path, result_path, problem):
    import os
    import json
    import torch

    assert problem in ['Skoda', 'PAMAP2', 'Oppotunity', 'USC_HAD', 'WISDM', 'WISDM2']
    config_path = os.path.join(root_path, result_path, 'configuration.json')

    with open(config_path) as f:
        config = json.load(f)
        config['problem'] = problem
        config['data_dir'] = os.path.join(root_path, 'Dataset/Benchmarks')
        config['output_dir'] = os.path.join(root_path, result_path)
        config['pred_dir'] = os.path.join(root_path, result_path, 'predictions')
        config['save_dir'] = os.path.join(root_path, result_path, 'checkpoints') # saved model path
        config['model_dir'] = os.path.join(config['save_dir'], f'{problem}_model_last.pth') # or f'{problem}_2_model_last.pth'
        config['tensorboard_dir'] = os.path.join(root_path, result_path, 'tb_summaries') # saved model path
        config['device'] = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    return config


def visualize_tsne(X, y, n_components=2, n_points=1000, problem="Skoda"):
    import os
    import random
    import pickle

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt 
    import seaborn as sns
    from sklearn.manifold import TSNE

    assert problem in ['Skoda', 'PAMAP2', 'Oppotunity', 'USC_HAD', 'WISDM', 'WISDM2']
    assert n_components in [2, 3]
    random.seed(1234)

    X_repr = X.cpu().detach().numpy()
    y_repr = y.cpu().detach().numpy()
    sample_idx = random.sample(range(len(y_repr)), min(n_points, len(y_repr)))

    tsne = TSNE(n_components=n_components) 
    X_tsne = tsne.fit_transform(X_repr[sample_idx, :]) 

    fig, ax = plt.subplots(figsize=(8, 8)) 
    data_tsne = np.vstack((X_tsne.T, y_repr[sample_idx])).T
    if n_components == 2:
        df_tsne = pd.DataFrame(data_tsne, columns=['dim1', 'dim2', 'class'])
        sns.scatterplot(data=df_tsne, hue='class', x='dim1', y='dim2')
    elif n_components == 3:
        sns.set_style("whitegrid", {'axes.grid' : False})
        df_tsne = pd.DataFrame(data_tsne, columns=['dim1', 'dim2', 'dim3', 'class']) 
        ax = plt.axes(projection="3d")
        ax.scatter(df_tsne['dim1'], df_tsne['dim2'], df_tsne['dim3'], c=df_tsne['class'], marker='o')
        # Experimental: save the figure to file, for interactive 3d view
        np.savez_compressed(f'{problem}_tsne.npz', data_tsne)
     
    fig.savefig(os.path.join(root_path, f'{problem}_tsne.png'))
    


problem = "Skoda"
root_path = "/home/shouzheyun/Series2Vec"
result_path = "Results/Pre_Training/Benchmarks/2024-09-14_18-12"
config = load_config_from_json(root_path, result_path, problem)
Data = data_loader(config)
model = Model_factory(config, Data)

# --------------------------------- Load Data ---------------------------------
train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

# --------------------------------- Load Model --------------------------------
SS_Encoder = load_model(model, model_path=config['model_dir'], optimizer=None)  # Loading the model
SS_Encoder.to(config['device'])
train_repr, train_labels = S2V_make_representation(SS_Encoder, train_loader)
# test_repr, test_labels = S2V_make_representation(SS_Encoder, test_loader)

# ------------------------------- Visualize Test ------------------------------
visualize_tsne(train_repr, train_labels, n_components=3, n_points=1000, problem=problem)
# visualize_tsne(test_repr.cpu().detach().numpy(), test_labels.cpu().detach().numpy())
