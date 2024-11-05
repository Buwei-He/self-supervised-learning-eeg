import os

from models.model_factory import Model_factory
from Dataset.dataloader import data_loader, dataset_class
from torch.utils.data import DataLoader
from Dataset.dataloader import dataset_class
from utils.utils import load_model
from models.Series2Vec.S2V_training import S2V_make_representation


def check_path_exists(path):
    if not os.path.exists(path):
        print(f'{path} not exists!\nPlease check \'root_path\' or \'result_path\' ...')


def load_config_from_json(root_path, result_path, problem):
    
    import json
    import torch

    assert problem in ['Skoda', 'PAMAP2', 'Oppotunity', 'USC_HAD', 'WISDM', 'WISDM2', 'EEG']
    config_path = os.path.join(root_path, result_path, 'configuration.json')
    with open(config_path) as f:
        config = json.load(f)
        config['problem'] = problem
        config['data_dir'] = os.path.join(root_path, 'Dataset/EEG')
        config['output_dir'] = os.path.join(root_path, result_path)
        config['pred_dir'] = os.path.join(root_path, result_path, 'predictions')
        config['save_dir'] = os.path.join(root_path, result_path, 'checkpoints') # saved model path
        config['model_dir'] = os.path.join(config['save_dir'], f'{problem}_model_last.pth') # or f'{problem}_2_model_last.pth'
        config['tensorboard_dir'] = os.path.join(root_path, result_path, 'tb_summaries') # saved model path
        config['device'] = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    return config


def visualize(X, y, reducer_type='tsne', problem='Skoda', n_components=2, n_points=1000, random_state=1234):
    import os
    import random

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt 
    import seaborn as sns

    from umap import UMAP # pip install umap-learn
    from sklearn.manifold import TSNE

    assert reducer_type in ['tsne', 'umap']
    assert problem in ['Skoda', 'PAMAP2', 'Oppotunity', 'USC_HAD', 'WISDM', 'WISDM2', 'EEG']
    assert n_components in [2, 3]
    random.seed(random_state)

    X_repr = X.cpu().detach().numpy()
    y_repr = y.cpu().detach().numpy()
    sample_idx = random.sample(range(len(y_repr)), min(n_points, len(y_repr)))

    if reducer_type == 'umap':
        reducer = UMAP(n_components=n_components) 
    elif reducer_type == 'tsne':
        reducer = TSNE(n_components=n_components)
    X_reduced = reducer.fit_transform(X_repr[sample_idx, :]) 

    fig, ax = plt.subplots(figsize=(8, 8)) 
    data_reduced = np.vstack((X_reduced.T, y_repr[sample_idx])).T
    if n_components == 2:
        df = pd.DataFrame(data_reduced, columns=['dim1', 'dim2', 'class'])
        sns.scatterplot(data=df, hue='class', x='dim1', y='dim2')
    elif n_components == 3:
        sns.set_style("whitegrid", {'axes.grid' : False})
        df = pd.DataFrame(data_reduced, columns=['dim1', 'dim2', 'dim3', 'class']) 
        ax = plt.axes(projection="3d")
        ax.scatter(df['dim1'], df['dim2'], df['dim3'], c=df['class'], marker='o')
        # Experimental: save the figure to file, for interactive 3d view
        np.savez_compressed(f'{problem}_{reducer_type}_{n_components}d.npz', data_reduced)
     
    fig.savefig(os.path.join(root_path, 'visualize', f'{problem}_{reducer_type}_{n_components}d.png'))

def visualize_nd(X, y, reducer_type='tsne', problem='Skoda', n_components=2, n_points=1000, random_state=1234):
    import os
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from umap import UMAP
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    assert reducer_type in ['tsne', 'umap', 'pca']
    assert problem in ['Skoda', 'PAMAP2', 'Oppotunity', 'USC_HAD', 'WISDM', 'WISDM2', 'EEG']
    assert n_components > 1, "n_components should be at least 2 for pairwise plotting."
    assert (reducer_type != 'tsne' or n_components < 4), "n_components should be inferior to 4 for tsne."
    random.seed(random_state)

    X_repr = X.cpu().detach().numpy()
    y_repr = y.cpu().detach().numpy()
    sample_idx = random.sample(range(len(y_repr)), min(n_points, len(y_repr)))

    # Initialize the dimensionality reducer
    if reducer_type == 'umap':
        reducer = UMAP(n_components=n_components)
    elif reducer_type == 'tsne':
        reducer = TSNE(n_components=n_components)
    elif reducer_type == 'pca':
        reducer = PCA(n_components=n_components)
    X_reduced = reducer.fit_transform(X_repr[sample_idx, :])

    # Create a DataFrame for easier plotting
    columns = [f'dim{i+1}' for i in range(n_components)]
    df = pd.DataFrame(X_reduced, columns=columns)
    df['class'] = y_repr[sample_idx]

    # Plot pairwise scatter plots in a matrix format
    num_plots = (n_components * (n_components - 1)) // 2
    fig, axes = plt.subplots(n_components - 1, n_components - 1, figsize=(3*n_components, 3*n_components), constrained_layout=True)
    fig.suptitle(f'{problem} - {reducer_type} - {n_components}D Pairwise Scatter Plots', fontsize=16)

    # Generate pairwise scatter plots
    for i in range(n_components - 1):
        for j in range(i + 1, n_components):
            ax = axes[i, j - 1] if n_components > 2 else axes
            sns.scatterplot(data=df, x=f'dim{i+1}', y=f'dim{j+1}', hue='class', ax=ax, legend=False)
            ax.set_title(f'Dim {i+1} vs Dim {j+1}')
            ax.set_xlabel(f'Dim {i+1}')
            ax.set_ylabel(f'Dim {j+1}')

    # Hide unused subplots
    for i in range(n_components - 1):
        for j in range(i):
            axes[i, j].axis('off')

    # Save the figure
    os.makedirs(os.path.join(root_path, 'visualize'), exist_ok=True)
    fig.savefig(os.path.join(root_path, 'visualize', f'{problem}_{reducer_type}_{n_components}D_pairwise.png'))


if __name__ == '__main__':
    problem = "EEG"
    root_path = "/home/buwei/dd2430/self-supervised-learning-eeg"
    result_path = "Results/Pre_Training/EEG/CF_0.75"
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

    # ------------------------------- Visualize Test ------------------------------
    # visualize(X=train_repr, 
    #         y=train_labels,
    #         reducer_type='tsne', # 'tsne', 'umap''
    #         problem=problem, 
    #         n_components=3, # 2, 3
    #         n_points=1000
    #         )
    
    visualize_nd(X=train_repr, 
            y=train_labels,
            reducer_type='umap', # 'tsne', 'umap', 'pca'
            problem=problem, 
            n_components=20, # 2, 3
            n_points=100
            )
