import os
import logging
import torch
import numpy as np
from collections import OrderedDict
import time
import shutil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.fft as fft
import torch.nn.functional as F
from utils import utils, analysis
from utils.eeg_utils import map_categories_to_numbers, map_numbers_to_categories

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.Series2Vec.soft_dtw_cuda import SoftDTW
from models.Series2Vec.fft_filter import filter_frequencies
from sklearn.linear_model import RidgeClassifier

logger = logging.getLogger('__main__')

# Define the TensorBoard log directory
log_dir = 'summary'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)  # Recursively delete the log directory and its contents
    print(f"Cleaned up TensorBoard directory: {log_dir}")
tensorboard_writer = SummaryWriter(log_dir=log_dir)

NEG_METRICS = {'loss'}  # metrics for which "better" is less


class BaseTrainer(object):

    def __init__(self, model, train_loader, test_loader, config, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat=False):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = config['device']
        self.optimizer = config['optimizer']
        self.loss_module = config['loss_module']
        self.k_fold = config['k_fold_cnt']
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()
        self.save_path = config['output_dir']
        self.problem = config['problem']
        self.seed = config['seed']
        self.batch_size = config['batch_size']

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):
        total_batches = len(self.dataloader)
        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class S2V_SS_Trainer(BaseTrainer):
    """
    Comment: 

    - Self-supervised learning (on training set);

    - Linear probing is integrated here: make representation (for training and testing set) 
    every 5 epochs, and output as "\<problem\>_linear_result.txt"
    
    -
    """
    def __init__(self, *args, **kwargs):
        super(S2V_SS_Trainer, self).__init__(*args, **kwargs)
        self.analyzer = analysis.Analyzer(print_conf_mat=False)
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)

    def ss_training_loss_fn(self, model, X, get_all_losses=False):
        '''
        Re-formatted version; \n
        Soft-DTW distance + smooth l1; \n
        Only for self-supervised pre-training ...
        '''
        Distance_out, Distance_out_f = model.Pretrain_forward(X.to(self.device))
        # Create a mask to select only the lower triangular values
        mask = torch.tril(torch.ones_like(Distance_out), diagonal=-1).bool()
        Distance_out = torch.masked_select(Distance_out, mask)
        Distance_out = Distance_normalizer(Distance_out)

        Distance_out_f = torch.masked_select(Distance_out_f, mask)
        Distance_out_f = Distance_normalizer(Distance_out_f)

        # X_smooth = moving_average_smooth(X, 3)

        Dtw_Distance = cuda_soft_DTW(self.sdtw, X, len(X))
        Dtw_Distance = Distance_normalizer(Dtw_Distance)

        # Euclidean_Distance = Euclidean_Dis(X, len(X))
        # Euclidean_Distance = Distance_normalizer(Euclidean_Distance)
        X_f = filter_frequencies(X)
        # X_f = fft.fft2(X, dim=(-2, -1))
        Euclidean_Distance_f = Euclidean_Dis(X_f, len(X_f))
        Euclidean_Distance_f = Distance_normalizer(Euclidean_Distance_f)

        # temporal_loss = torch.nn.functional.mse_loss(Distance_out, Dtw_Distance)
        # frequency_loss = torch.nn.functional.mse_loss(Distance_out_f, Euclidean_Distance_f)
        temporal_loss = F.smooth_l1_loss(Distance_out, Dtw_Distance)
        frequency_loss = F.smooth_l1_loss(Distance_out_f, Euclidean_Distance_f)

        total_loss = temporal_loss + frequency_loss

        if get_all_losses:
            return total_loss, temporal_loss, frequency_loss
        else:
            return total_loss


    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        train_time_loss = 0
        train_freq_loss = 0
        total_samples = 0  # total samples in epoch
        for i, batch in enumerate(self.train_loader):
            X, _, IDs = batch
            # No need to train on uncomplete batches
            if X.shape[0] < self.batch_size:
                continue
            else:
                total_loss, time_loss, freq_loss = self.ss_training_loss_fn(self.model, X, get_all_losses=True)
                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                total_loss.backward()

                # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                self.optimizer.step()

                with torch.no_grad():
                    total_samples += 1
                    epoch_loss += total_loss.item()
                    train_time_loss += time_loss.item()
                    train_freq_loss += freq_loss.item()

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        train_time_loss = train_time_loss / total_samples
        train_freq_loss = train_freq_loss / total_samples
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        # if (epoch_num+1) % 5 == 0:
        self.evaluate(epoch_num=epoch_num, epoch_train_loss=[epoch_loss, time_loss, freq_loss])

        return self.epoch_metrics


    def evaluate(self, epoch_num=None, epoch_train_loss=None):
        '''
        There is no evaluate process in ss_trainer, implemented to get Soft-DTW related loss on test set only.
        '''
        assert(self.test_loader is not None)

        self.model = self.model.eval()

        # downstream task (classification) analysis: accuracy of each class / avg. of all classes
        train_repr, train_labels, train_info = S2V_make_representation(self.model, self.train_loader)
        test_repr, test_labels, test_info = S2V_make_representation(self.model, self.test_loader)

        clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())

        train_y_hat = clf.predict(train_repr.cpu().detach().numpy())
        test_y_hat = clf.predict(test_repr.cpu().detach().numpy())

        train_acc, train_class_acc = analysis.subject_wise_analysis(
                                    y_true=train_labels.cpu().detach().numpy(), 
                                    y_pred=train_y_hat, 
                                    subject_info=train_info,
                                    epoch_num=epoch_num,
                                    k_fold=self.k_fold,
                                    result_path=self.save_path)

        test_acc, test_class_acc = analysis.subject_wise_analysis(
                                    y_true=test_labels.cpu().detach().numpy(), 
                                    y_pred=test_y_hat, 
                                    subject_info=test_info,
                                    epoch_num=epoch_num,
                                    dataset='test',
                                    k_fold=self.k_fold,
                                    result_path=self.save_path)

        result_file = open(f'{self.save_path}/{self.problem}_linear_result.txt', 'a+')
        
        # Add to tensorboard
        unique_classes = map_numbers_to_categories(np.unique(test_labels.cpu().detach().numpy()))
        class_accuracies = {}
        keys_str = ''
        values_str = ''
        for cls in unique_classes:
            cls_acc = (test_class_acc[cls].values)[0]
            class_accuracies[f'test_class_{cls}'] = cls_acc
            keys_str += f', test_acc_{cls}'
            values_str += ', {0:.8f}'.format(cls_acc)

        for cls in unique_classes:
            cls_acc = (train_class_acc[cls].values)[0]
            class_accuracies[f'train_class_{cls}'] = cls_acc
            keys_str += f', train_acc_{cls}'
            values_str += ', {0:.8f}'.format(cls_acc)


        # Log class-wise accuracies to TensorBoard
        class_accuracies[f'test_class_all'] = test_acc
        class_accuracies[f'train_class_all'] = train_acc
        tensorboard_writer.add_scalars(f'acc', class_accuracies, epoch_num)


        # Soft-DTW related loss on test dataset
        epoch_test_loss = np.zeros(3)
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.test_loader):
            X, _, IDs = batch
            total_loss, time_loss, freq_loss = self.ss_training_loss_fn(self.model, X, get_all_losses=True)
            total_samples += 1
            epoch_test_loss[0] += total_loss.item()
            epoch_test_loss[1] += time_loss.item()
            epoch_test_loss[2] += freq_loss.item()

        epoch_test_loss = epoch_test_loss / total_samples

        tensorboard_writer.add_scalars(f'loss', {'test_total':epoch_test_loss[0]}, epoch_num)
        tensorboard_writer.add_scalars(f'loss', {'test_time':epoch_test_loss[1]}, epoch_num)
        tensorboard_writer.add_scalars(f'loss', {'test_freq':epoch_test_loss[2]}, epoch_num)
        tensorboard_writer.add_scalars(f'loss', {'train_total':epoch_train_loss[0]}, epoch_num) 
        tensorboard_writer.add_scalars(f'loss', {'train_time':epoch_train_loss[1]}, epoch_num)
        tensorboard_writer.add_scalars(f'loss', {'train_freq':epoch_train_loss[2]}, epoch_num)

        if epoch_num == 0:
            print(f'#, train_loss, test_loss, train_acc_all, test_acc_all{keys_str}', file=result_file)

        print('{0}, {1:.8f}, {2:.8f}, {3:.8f}, {4:.8f}{5}'.format(
            epoch_num, epoch_train_loss[0], epoch_test_loss[0], train_acc, test_acc, values_str
            ), 
            file=result_file)
        result_file.close()
            

class S2V_S_Trainer(BaseTrainer):
    """
    Comment: 

    - Supervised learning (on training set);
    
    - Evaluate: calc. loss, and use softmax to get estimated probability of classes;

    """
    def __init__(self, *args, **kwargs):
        super(S2V_S_Trainer, self).__init__(*args, **kwargs)
        self.analyzer = analysis.Analyzer(print_conf_mat=False)
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)

    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        for i, batch in enumerate(self.train_loader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            total_loss = batch_loss / len(loss)  # mean loss (over samples)

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            with torch.no_grad():
                total_samples += 1
                epoch_loss += total_loss.item()

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.train_loader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()

            per_batch['targets'].append(targets.cpu().numpy())
            predictions = predictions.detach()
            per_batch['predictions'].append(predictions.cpu().numpy())
            loss = loss.detach()
            per_batch['metrics'].append([loss.cpu().numpy()])
            per_batch['IDs'].append(IDs)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(predictions,
                                            dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)

        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes

        # # CUSTOM downstream task (classification) analysis
        # train_repr, train_labels, train_info = S2V_make_representation(self.model, self.train_loader)
        # test_repr, test_labels, test_info = S2V_make_representation(self.model, self.test_loader)
        # clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
        # train_y_hat = clf.predict(train_repr.cpu().detach().numpy())
        # test_y_hat = clf.predict(test_repr.cpu().detach().numpy())
        # train_acc, train_class_acc = analysis.subject_wise_analysis(
        #                             y_true=train_labels.cpu().detach().numpy(), 
        #                             y_pred=train_y_hat, 
        #                             subject_info=train_info,
        #                             epoch_num=epoch_num,
        #                             k_fold=self.k_fold,
        #                             result_path=self.save_path)

        # test_acc, test_class_acc = analysis.subject_wise_analysis(
        #                             y_true=test_labels.cpu().detach().numpy(), 
        #                             y_pred=test_y_hat, 
        #                             subject_info=test_info,
        #                             epoch_num=epoch_num,
        #                             k_fold=self.k_fold,
        #                             result_path=self.save_path)

        return self.epoch_metrics, metrics_dict


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    with torch.no_grad():
        aggr_metrics, ConfMat = val_evaluator.evaluate(epoch, keep_all=True)

    print()
    print_str = 'Validation Summary: '
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

    return aggr_metrics, best_metrics, best_value


def S_train_runner(config, model, trainer, evaluator, path):
    epochs = config['epochs'] + config['epochs_ft']
    optimizer = config['optimizer']
    loss_module = config['loss_module']
    start_epoch = config['epochs']
    total_start_time = time.time()
    # tensorboard_writer = SummaryWriter('summary')
    best_value = 1e16
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
    save_best_model = utils.SaveBestACCModel()

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        aggr_metrics_val, best_metrics, best_value = validate(evaluator, tensorboard_writer, config, best_metrics,
                                                                best_value, epoch)
        save_best_model(aggr_metrics_val['accuracy'], epoch, model, optimizer, loss_module, path)
        metrics_names, metrics_values = zip(*aggr_metrics_train.items())
        metrics.append(list(metrics_values))

        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    return


def SS_train_runner(config, model, trainer, path):
    epochs = config['epochs']
    optimizer = config['optimizer']
    loss_module = config['loss_module']
    start_epoch = 0
    total_start_time = time.time()
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    save_best_model = utils.SaveBestModel()
    Total_loss = []

    # Evaluate before training
    # Soft-DTW related loss on test dataset
    epoch_train_loss = np.zeros(3)
    total_samples = 0  # total samples in epoch

    for i, batch in enumerate(trainer.train_loader):
        X, _, IDs = batch
        total_loss, time_loss, freq_loss = trainer.ss_training_loss_fn(model, X, get_all_losses=True)
        total_samples += 1
        epoch_train_loss[0] += total_loss.item()
        epoch_train_loss[1] += time_loss.item()
        epoch_train_loss[2] += freq_loss.item()

    epoch_train_loss = epoch_train_loss / total_samples
    trainer.evaluate(epoch_num=0, epoch_train_loss=epoch_train_loss)

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        save_best_model(aggr_metrics_train['loss'], epoch, model, optimizer, loss_module, path)
        metrics_names, metrics_values = zip(*aggr_metrics_train.items())
        metrics.append(list(metrics_values))
        Total_loss.append(aggr_metrics_train['loss'])
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            if k != 'epoch':
                tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
    # plot_loss(Total_loss,Time_loss,Freq_loss)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    return


def cuda_soft_DTW(sdtw, X, size):
    # index = list(product([*range(size)], repeat=2))
    index = generate_list(size-1)
    combination1 = X[[i[0] for i in index]].to('cuda')
    combination2 = X[[i[1] for i in index]].to('cuda')
    Dtw_Distance = sdtw(combination1, combination2)
    return Dtw_Distance


def Euclidean_Dis(X, size):
    index = generate_list(size - 1)
    combination1 = X[[i[0] for i in index]].to('cuda')
    combination2 = X[[i[1] for i in index]].to('cuda')
    combination1_flat = combination1.view(combination1.size(0), -1)
    combination2_flat = combination2.view(combination2.size(0), -1)
    distances = torch.norm(combination1_flat - combination2_flat, dim=1)
    return distances


def generate_list(num):
    result = []
    for i in range(1, num+1):
        for j in range(0, i):
            result.append((i, j))
    return result


def Distance_normalizer(distance):

    if len(distance) == 1:
        Normal_distance = distance/distance
    else:
        min_val = torch.min(distance)
        max_val = torch.max(distance)

        # Normalize the distances between 0 and 1
        Normal_distance = (distance - min_val) / (max_val - min_val)
    return Normal_distance


def S2V_make_representation(model, data):
    out = []
    labels = []
    meta_infos = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data):
            X, targets, IDs = batch
            rep = model.linear_prob(X.to('cuda'))
            out.append(rep)
            labels.append(targets)
            if data.dataset.meta_info is not None:
                meta_infos.append(data.dataset.meta_info[IDs])
        out = torch.cat(out, dim=0)
        labels = torch.cat(labels, dim=0)
        if data.dataset.meta_info is not None:
            meta_infos = np.concatenate(meta_infos, axis=0)
            return out, labels, meta_infos
    return out, labels

#logistic regression
def fit_lr(features, y, MAX_SAMPLES=100000, seed=1234):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=seed, stratify=y
        )
        features = split[0]
        y = split[2]

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=seed,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe

#####
def fit_RidgeClassifier(features, y, MAX_SAMPLES=100000, seed=1234):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=seed, stratify=y
        )
        features = split[0]
        y = split[2]

    pipe = make_pipeline(
        StandardScaler(),
        RidgeClassifier()
    )
    pipe.fit(features, y)
    return pipe


def moving_average_smooth(data, window_size):
    """
    Smooth the input multivariate time series using a moving average separately for each channel.

    Parameters:
    - data (torch.Tensor): Input time series data of shape (sequence_length, num_variables).
    - window_size (int): Size of the moving average window.

    Returns:
    - smoothed_data (torch.Tensor): Smoothed time series data of the same shape as input.
    """
    num_variables = data.shape[1]

    # Use a 1D convolution with a uniform filter to perform the moving average for each channel
    smoothed_data = torch.zeros_like(data)
    for i in range(num_variables):
        kernel = torch.ones(window_size) / window_size
        kernel = kernel.view(1, 1, window_size)

        # Use padding to handle the edges of the time series
        padding = (window_size - 1) // 2
        channel_data = data[:, i:i + 1, :]
        smoothed_channel = F.conv1d(channel_data, kernel, padding=padding)
        smoothed_data[:, i:i + 1, :] = smoothed_channel

    return smoothed_data


def filter_low_frequencies_batch(batch_data, threshold_freq=40):
    # Convert to PyTorch tensor
    batch_data_tensor = torch.tensor(batch_data, dtype=torch.float32)

    # Apply 2D FFT
    fft_result = fft.fft2(batch_data_tensor, dim=(-2, -1))

    # Shift zero frequency component to the center
    fft_shifted = fft.fftshift(fft_result, dim=(-2, -1))

    # Get frequency components
    rows, cols = batch_data.shape[-2], batch_data.shape[-1]
    # Calculate corresponding frequencies in Hz
    freq_rows_hz = fft.fftfreq(rows, d=1.0)
    freq_cols_hz = fft.fftfreq(cols, d=1.0)

    # Create meshgrid for frequency components
    freq_rows_mesh, freq_cols_mesh = torch.meshgrid(torch.tensor(freq_rows_hz), torch.tensor(freq_cols_hz))

    # Filter frequencies lower than the threshold
    mask = (torch.abs(freq_rows_mesh) < threshold_freq) & (torch.abs(freq_cols_mesh) < threshold_freq)
    fft_shifted_filtered = fft_shifted * mask

    # Inverse FFT to get filtered signal
    filtered_data = fft.ifft2(fft.ifftshift(fft_shifted_filtered), dim=(-2, -1)).real

    return filtered_data


