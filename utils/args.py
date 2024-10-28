import argparse
import os
import json
from datetime import datetime
import torch
import logging
from typing import List

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()


def Initialization(args):
    """
            Input:
                args: arguments object from argparse
            Returns:
                config: configuration dictionary
    """

    config = args.args.__dict__  # configuration dictionary
    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, config['Training_mode'], config['dataset'],
                              initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    config['output_dir'] = output_dir
    config['data_dir'] = os.getcwd() + '/Dataset/' + config['dataset']
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)
    logger.info("Stored configuration file in '{}'".format(output_dir))
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
    config['device'] = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(config['device']))
    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


# -------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--dataset', default='EEG', choices={'Benchmarks', 'UEA', 'UCR', 'EEG'})
parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=True, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of the train-set to be used as validation")
parser.add_argument('--test_ratio', type=float, default=0.2, help="Proportion of the dataset that is kept for testing (neither in train nor validation)")
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
parser.add_argument('--duration', type=int, default=5, help='Duration (in s) for one epoch of data')
parser.add_argument('--sample_rate', type=int, default=100, help='Resampling rate of EEG signal')
parser.add_argument('--overlap_ratio', type=float, default=0, help="Overlap ratio of epochs")
parser.add_argument('--channels', default=['Cz','Pz'], help='EEG channels to consider, "all" for all channels')
parser.add_argument('--MMSE_max_A', type=int, default=30, help='Max MMSE score for AD subjects')
parser.add_argument('--MMSE_max_F', type=int, default=30, help='Max MMSE score for FTD scubjects')
parser.add_argument('--classes', type=List[str], default=['A','C','F'], help='Classes to use in EEG problem')
parser.add_argument('--create_data', type=bool, default=True, help='Only for EEG. Whether to create the datasets from the args provided. If False supposes the existence of a EEG.npy file in the dataset folder but parameters may be inconsistent with config.')
parser.add_argument('--max_train_samples', type=int, default=1000, help='Maximum number of training samples to use for each class. If None, uses the maximum.')
# ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
parser.add_argument('--Training_mode', default='Pre_Training', choices={'Pre_Training', 'Linear_Probing', 'Supervised'})
parser.add_argument('--Model_Type', default=['Series2Vec'], choices={'Series2Vec', 'TS_TCC', 'TF_C'})
parser.add_argument('--layers', type=int, default=4, help="Number of layers for input conv encoders")
parser.add_argument('--emb_size', type=int, default=12, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of dense feedforward part of transformer layer')
parser.add_argument('--rep_size', type=int, default=512, help='Representation dimension')
parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')
# -------------------------------------Training Parameters/ Hyper-Parameters -----------------------------------------
parser.add_argument('--epochs', type=int, default=5, help='Number of pre-training epochs')
parser.add_argument('--epochs_ft', type=int, default=10, help='Number of fine-tuning epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.01, help='Dropout regularization ratio')
parser.add_argument('--val_interval', type=int, default=1, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='loss',
                    help='Metric used for defining best epoch')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
args = parser.parse_args()





