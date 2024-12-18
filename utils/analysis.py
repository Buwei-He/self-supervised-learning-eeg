"""
Collection of functions which enable the evaluation of a classifier's performance,
by showing confusion matrix, accuracy, recall, precision etc.
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt
from sklearn import metrics
from tabulate import tabulate
import math
import logging
from datetime import datetime
from utils.eeg_utils import map_numbers_to_categories

def subject_wise_analysis(y_true, y_pred, subject_info, result_path='./',
                          k_fold=0, epoch_num='final',dataset='train',
                          export_analysis = True,
                          export_subject_accuracy = True,
                          export_subject_conf_matrix = True,
                          export_subject_conf_image = False,
                          ):
    """
    Use np.load(path, allow_pickle=True).item() to recover analysis result.
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if k_fold > 0:
        result_path = os.path.join(result_path, f'k_fold_{k_fold}')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Create a DataFrame to group by subject ID
    data_df = pd.DataFrame({'subject_id': subject_info[:, 0].astype(str), 
                            'y_true': y_true, 
                            'y_pred': y_pred, 
                            'start_time': subject_info[:, 1]})

    # Initialize dictionaries to store each subject's confusion matrix and metrics
    subject_conf_matrices = {}
    labels = np.sort(np.unique(y_true))
    label_names = map_numbers_to_categories(labels)
    label_name_mapping = {label_names[i]: labels[i] for i in range(len(labels))}

    # Calculate confusion matrix and metrics for each subject
    for subject_id, group in data_df.groupby('subject_id'):
        
        y_true_subject = map_numbers_to_categories(group['y_true'].values)
        y_pred_subject = map_numbers_to_categories(group['y_pred'].values)
        
        # Generate confusion matrix for this subject
        conf_matrix = confusion_matrix(y_true_subject, y_pred_subject, labels=label_names)
        assert np.all(y_true_subject == y_true_subject[0])
        subject_conf_matrices[subject_id] = conf_matrix

    # Initialize variables
    class_counts = None  # To store sum of instances for each class across all subjects
    true_positives = None  # To store sum of true positives for each class across all subjects
    total_correct = 0  # To count total correct predictions across all subjects
    total_predictions = 0  # To count all predictions across all subjects

    # To store results per subject and class
    subject_class_accuracies = {}
    
    for subject_id, matrix in subject_conf_matrices.items():
        subject_true_positives = np.diag(matrix)
        class_totals = matrix.sum(axis=1)
        subject_class_accuracy = subject_true_positives / class_totals
        
        # Store individual subject-class accuracies
        subject_class_accuracies[subject_id] = {
            'vote_percentage': np.nanmax(subject_class_accuracy),
            'true_label': label_names[np.nanargmax(subject_class_accuracy)],
            'vote_label': label_names[np.argmax(np.sum(matrix, axis=0))],
            'correct_vote': np.max(subject_true_positives) == np.max(matrix),
        }

        # Aggregate true positives and class totals across all subjects
        if true_positives is None:
            true_positives = subject_true_positives
            class_counts = class_totals
        else:
            true_positives = true_positives + subject_true_positives
            class_counts += class_totals

        # Update overall accuracy counters
        total_correct += subject_true_positives.sum()
        total_predictions += matrix.sum()


    class_accuracies = true_positives / class_counts
    weighted_avg_accuracy = total_correct / total_predictions

    df = pd.DataFrame(subject_class_accuracies).T
    print(f'Analysis: k_fold = {k_fold}, epoch = {epoch_num}, on {dataset} set')
    print("Class-wise accuracy across all subjects:", class_accuracies)
    print("Weighted avg. accuracy:", weighted_avg_accuracy)

    voting_conf_matrix = confusion_matrix(df['true_label'].values, df['vote_label'].values, labels=label_names)
    voting_norm_conf_matrix = np.nan_to_num(voting_conf_matrix / voting_conf_matrix.sum(axis=1)[:, np.newaxis], nan=0)
    voting_acc = np.diag(voting_norm_conf_matrix)
    voting_avg_acc = np.mean(df['correct_vote'].values.astype(float))
    # update: convert to dataframe for compatibility with binary classification
    voting_conf_df = pd.DataFrame(data=voting_conf_matrix, columns=label_names, index=label_names)
    voting_norm_conf_df = pd.DataFrame(data=voting_norm_conf_matrix, columns=label_names, index=label_names)
    voting_acc_df = pd.DataFrame(data=voting_acc[np.newaxis, :], columns=label_names, index=['acc.'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print("Voting accuracy:\n", voting_acc_df)
        print("Voting confusion matrix:\n", voting_conf_df)
        print("Voting confusion matrix, normalized:\n", voting_norm_conf_df)

    # TODO: save the confusion matrix as txt file
    analysis = {
        'k_fold': k_fold,
        'epoch_num': epoch_num,
        'dataset': dataset,
        'voting_accuracy': voting_avg_acc,
        'voting_class_accuracy': voting_acc,
        'voting_conf_matrix': voting_conf_matrix,
        'voting_norm_conf_matrix': voting_norm_conf_matrix,
    }

    if export_subject_accuracy:
        analysis['subject_class_accuracy'] = subject_class_accuracies
    if export_subject_conf_matrix:
        analysis['subject_conf_matrix'] = subject_conf_matrices

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(f"subject-wise, per-class accuracy at epoch {epoch_num}:\n", df)
    
    if export_analysis:
        output_file = os.path.join(result_path, f'analysis_{dataset}_epoch_{epoch_num}')
        np.save(output_file, analysis, allow_pickle=True)

    if export_subject_conf_image:
        # Set the number of rows and columns for the subplot grid
        num_subjects = len(subject_conf_matrices)
        cols = 6  # Adjust based on how many matrices you want per row
        rows = (num_subjects + cols - 1) // cols  # Calculate required rows

        # Set up the figure size and style
        plt.figure(figsize=(2.4 * cols, 2.4 * rows))
        plt.suptitle(f"Confusion Matrices for Each subject, Epoch {epoch_num}", fontsize=16)

        # Generate a confusion matrix subplot for each subject
        for idx, (subject_id, matrix) in enumerate(subject_conf_matrices.items(), start=1):
            plt.subplot(rows, cols, idx)
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'sub-0{subject_id}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

        # Adjust layout and save the figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
        output_file = os.path.join(result_path, f"conf_matrices_epoch_{epoch_num}.png")
        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"Confusion matrices saved as {output_file}")
    return voting_avg_acc, voting_acc_df


def acc_top_k(predictions, y_true):
    """Accuracy when allowing for correct class being in the top k predictions.

    Arguments:
        predictions: (N_samples, k) array of top class indices (pre-sorted class indices based on score) per sample
        y_true: N_samples 1D-array of ground truth labels (integer indices)
    Returns:
        length k 1D-array of accuracy when allowing for correct class being in top 1, 2, ... k predictions"""

    y_true = y_true[:, np.newaxis]

    # Create upper triangular matrix of ones, to be used in construction of V
    building_blocks = np.zeros((predictions.shape[1], predictions.shape[1]))
    building_blocks[np.triu_indices(predictions.shape[1])] = 1

    # A matrix of the same shape as predictions. For each sample, the index corresponding
    # to a correct prediction is 1, as well as all following indices.
    # Example: y_true = [1,0], predictions = [[1 5 4],[2 0 3]]. Then: V = [[1 1 1],[0 1 1]]
    V = np.zeros_like(predictions, dtype=int)  # validity matrix
    sample_ind, rank_ind = np.where(predictions == y_true)

    V[sample_ind, :] = building_blocks[rank_ind, :]

    return np.mean(V, axis=0)


def accuracy(y_pred, y_true, excluded_labels=None):
    """A simple accuracy calculator, which can ignore labels specified in a list"""

    if excluded_labels is None:
        return np.mean(y_pred == y_true)
    else:
        included = (y_pred != excluded_labels[0]) & (y_true != excluded_labels[0])
        # The following extra check (rather than initializing with an array of ones)
        # is done because a single excluded label is the most common case
        if len(excluded_labels) > 1:
            for label in excluded_labels[1:]:
                included &= (y_pred != label) & (y_true != label)

        return np.mean(y_pred[included] == y_true[included])


def precision(y_true, y_pred, label):
    """Returns precision for the specified class index"""

    predicted_in_C = (y_pred == label)
    num_pred_in_C = np.sum(predicted_in_C)
    if num_pred_in_C == 0:
        return 0
    return np.sum(y_true[predicted_in_C] == label) / num_pred_in_C


def recall(y_true, y_pred, label):
    """Returns recall for the specified class index"""

    truly_in_C = (y_true == label)
    num_truly_in_C = np.sum(truly_in_C)
    if num_truly_in_C == 0:
        return 0  # or NaN?
    return np.sum(y_pred[truly_in_C] == label) / num_truly_in_C


def limiter(metric_functions, y_true, y_pred, y_scores, score_thr, label):
    """Wraps a list of metric functions, i.e precison or recall, by ingoring predictions under the
    specified threshold for a specific class.
    """

    ltd_pred = np.copy(y_pred)
    ltd_pred[(ltd_pred == label) & (y_scores < score_thr)] = -1

    output = [func(y_true, ltd_pred, label) for func in metric_functions]

    return output


def prec_rec_parametrized_by_thr(y_true, y_pred, y_scores, label, Npoints, min_score=None, max_score=None):
    """Returns an array showing for a specified class of interest, how precision and recall change as a function of
        the score threshold (parameter).

    Input:
        y_true: 1D array of true labels (class indices)
        y_pred: 1D array of predicted labels (class indices)
        y_scores: 1D array of scores corresponding to predictions in y_pred
        label: class label of interest
        Npoints: number of score threshold points. Defines "resolution" of the parameter (score threshold)
        min_score, max_score: if specified, they impose lower and upper bound limits for the parameter (score thr.)
    Output:
        prec_rec: ndarray of shape (Npoints, 2), containing a precision (column 0) and recall (column 1) value for each
            score threshold value
    """

    if (min_score is None) or (max_score is None):
        predicted_in_C = (y_pred == label)
        min_score = 0.99 * np.amin(y_scores[predicted_in_C])  # guarantees that all predictions are kept
        max_score = 1.01 * np.amax(y_scores[predicted_in_C])  # guarantees that no prediction is kept

    grid = np.linspace(min_score, max_score, Npoints)

    measure = lambda x: limiter([precision, recall], y_true, y_pred, y_scores, x, label)

    return np.array(map(measure, grid)), grid


def plot_prec_vs_rec(score_grid, rec, prec, prec_requirement=None, thr_opt=None, title=None, show=True, save_as=None):
    """Plots a figure depicting precision and recall as a function of the score threshold.
    Optionally also depicts an imposed precision requirement and a chosen score threshold value."""

    if not (thr_opt is None):
        thr_opt = thr_opt if not (math.isinf(thr_opt)) else None

    plt.figure()
    if title:
        plt.suptitle(title)

    # Recall and Precision vs. Score Threshold
    plt.subplot(211)
    l_rec, = plt.plot(score_grid, rec, '.-')

    plt.hold(True)
    l_prec, = plt.plot(score_grid, prec, 'g.-')
    plt.ylim((0, 1.01))
    plt.xlabel('score threshold')

    legend_lines = [l_rec, l_prec]
    legend_labels = ['recall', 'precision']

    if prec_requirement:
        l_prec_req = plt.axhline(prec_requirement, color='r', linestyle='--')
        legend_lines.append(l_prec_req)
        legend_labels.append('prec. req.')

    if not (thr_opt is None):
        l_score_thr = plt.axvline(thr_opt, color='r')
        legend_lines.append(l_score_thr)
        legend_labels.append('opt. thr.')

    plt.legend(legend_lines, legend_labels, loc='lower right', fontsize=10)

    # Recall vs. Precision
    plt.subplot(212)
    plt.plot(prec, rec, '.-')

    plt.ylim((0, 1.01))
    plt.xlim((0, 1.01))
    plt.ylabel('recall')
    plt.xlabel('precision')

    if prec_requirement:
        l_prec_req = plt.axvline(prec_requirement, color='r', linestyle='--')
        plt.legend([l_prec_req], ['precision req.'], loc='lower left', fontsize=10)

    if save_as:
        plt.savefig(save_as, bbox_inches='tight', format='pdf')

    if show:
        plt.tight_layout()
        plt.show(block=False)


def plot_confusion_matrix(ConfMat, label_strings=None, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    """Plot confusion matrix in a separate window"""
    plt.imshow(ConfMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if label_strings:
        tick_marks = np.arange(len(label_strings))
        plt.xticks(tick_marks, label_strings, rotation=90)
        plt.yticks(tick_marks, label_strings)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def print_confusion_matrix(ConfMat, label_strings=None, title='Confusion matrix'):
    """Print confusion matrix as text to terminal"""

    if label_strings is None:
        label_strings = ConfMat.shape[0] * ['']

    print(title)
    print(len(title) * '-')
    # Make printable matrix:
    print_mat = []
    for i, row in enumerate(ConfMat):
        print_mat.append([label_strings[i]] + list(row))
    print(tabulate(print_mat, headers=['True\Pred'] + label_strings, tablefmt='orgtbl'))



class Analyzer(object):

    def __init__(self, maxcharlength=35, plot=False, print_conf_mat=False, output_filepath=None):

        self.maxcharlength = maxcharlength
        self.plot = plot
        self.print_conf_mat = print_conf_mat

        # create logger
        self.logID = str(
            datetime.now())  # this is to enable individual logging configuration between different instances
        self.logger = logging.getLogger(self.logID)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')

        # create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if output_filepath:
            # create file handler
            fh = logging.FileHandler(output_filepath)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def show_acc_top_k_improvement(self, y_pred, y_true, k=5, inp='scores'):
        """
        Show how accuracy improves when considering the event of the correct label being among the top k predictions as a successful prediction
        Arguments:
            k: integer k mentioned above
            inp: string, one of 'scores' or 'indices', defining assumptions for `y_pred`, see below
            y_pred: If inp is 'indices', then this is a (N_samples, k) array of top class indices (pre-sorted class indices based on score) per sample
                If inp is 'scores', then this is assummed to be a (N_samples, C) array of class scores per sample, where C is the number of classes
            y_true: (N_samples,) 1D numpy array of ground truth labels (integer indices)
        """

        print('How accuracy improves when allowing correct result being in the top 1, 2, ..., k predictions:\n')

        if inp == 'scores':
            predictions = np.argsort(y_pred, axis=1)[:, ::-1]  # sort in descending order
        else:
            predictions = y_pred

        predictions = predictions[:, :min(k, predictions.shape[1])]  # take top k

        accuracy_per_rank = acc_top_k(predictions, y_true)

        row1 = ['k'] + range(1, len(accuracy_per_rank) + 1)
        row2 = ['Accuracy'] + list(accuracy_per_rank)
        print(tabulate([row1, row2], tablefmt='orgtbl'))

        if self.plot:
            from matplotlib.ticker import MaxNLocator

            ax = plt.figure().gca()
            plt.plot(np.arange(1, k + 1, dtype=int), accuracy_per_rank, '.-')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('Number of allowed predictions (k)')
            plt.ylabel('Cumulative accuracy\n(prob. of correct result being in top k pred.)')
            plt.title('Cumulative Accuracy vs Number of allowed predictions')

            plt.show(block=False)

        return accuracy_per_rank

    def generate_classification_report(self, digits=3, number_of_thieves=2, maxcharlength=35):
        """
        Returns a string of a report for given metric arrays (array length equals the number of classes).
        Called internally by `analyze_classification`.
            digits: number of digits after . for displaying results
            number_of_thieves: number of biggest thieves to report
            maxcharlength: max. number of characters to use when displaying thief names
        """

        relative_freq = self.support / np.sum(self.support)  # relative frequencies of each class in the true lables
        sorted_class_indices = np.argsort(relative_freq)[
                               ::-1]  # sort by "importance" of classes (i.e. occurance frequency)

        last_line_heading = 'avg / total'

        width = max(len(cn) for cn in self.existing_class_names)
        width = max(width, len(last_line_heading), digits)

        headers = ["precision", "recall", "f1-score", "rel. freq.", "abs. freq.", "biggest thieves"]
        fmt = '%% %ds' % width  # first column: class name
        fmt += '  '
        fmt += ' '.join(['% 10s' for _ in headers[:-1]])
        fmt += '|\t % 5s'
        fmt += '\n'

        headers = [""] + headers
        report = fmt % tuple(headers)
        report += '\n'

        for i in sorted_class_indices:
            values = [self.existing_class_names[i]]
            for v in (self.precision[i], self.recall[i], self.f1[i],
                      relative_freq[i]):  # v is NOT a tuple, just goes through this list 1 el. at a time
                values += ["{0:0.{1}f}".format(v, digits)]
            values += ["{}".format(self.support[i])]
            thieves = np.argsort(self.ConfMatrix_normalized_row[i, :])[::-1][
                      :number_of_thieves + 1]  # other class indices "stealing" from class. May still contain self
            thieves = thieves[thieves != i]  # exclude self at this point
            steal_ratio = self.ConfMatrix_normalized_row[i, thieves]
            thieves_names = [
                self.existing_class_names[thief][:min(maxcharlength, len(self.existing_class_names[thief]))] for thief
                in thieves]  # a little inefficient but inconsequential
            string_about_stealing = ""
            for j in range(len(thieves)):
                string_about_stealing += "{0}: {1:.3f},\t".format(thieves_names[j], steal_ratio[j])
            values += [string_about_stealing]

            report += fmt % tuple(values)

        report += '\n' + 100 * '-' + '\n'

        # compute averages/sums
        values = [last_line_heading]
        for v in (np.average(self.precision, weights=relative_freq),
                  np.average(self.recall, weights=relative_freq),
                  np.average(self.f1, weights=relative_freq)):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ['{0}'.format(np.sum(relative_freq))]
        values += ['{0}'.format(np.sum(self.support))]
        values += ['']

        # make last ("Total") line for report
        report += fmt % tuple(values)

        return report

    def get_avg_prec_recall(self, ConfMatrix, existing_class_names, excluded_classes=None):
        """Get average recall and precision, using class frequencies as weights, optionally excluding
        specified classes"""

        class2ind = dict(zip(existing_class_names, range(len(existing_class_names))))
        included_c = np.full(len(existing_class_names), 1, dtype=bool)

        if not (excluded_classes is None):
            excl_ind = [class2ind[excl_class] for excl_class in excluded_classes]
            included_c[excl_ind] = False

        pred_per_class = np.sum(ConfMatrix, axis=0)
        nonzero_pred = (pred_per_class > 0)

        included = included_c & nonzero_pred
        support = np.sum(ConfMatrix, axis=1)
        weights = support[included] / np.sum(support[included])

        prec = np.diag(ConfMatrix[included, :][:, included]) / pred_per_class[included]
        prec_avg = np.dot(weights, prec)

        # rec = np.diag(ConfMatrix[included_c,:][:,included_c])/support[included_c]
        rec_avg = np.trace(ConfMatrix[included_c, :][:, included_c]) / np.sum(support[included_c])

        return prec_avg, rec_avg

    def prec_rec_histogram(self, precision, recall, binedges=None):
        """Make a histogram with the distribution of classes with respect to precision and recall
        """

        if binedges is None:
            binedges = np.concatenate((np.arange(0, 0.6, 0.2), np.arange(0.6, 1.01, 0.1)), axis=0)
            binedges = np.append(binedges, binedges[-1] + 0.1)  # add 1 extra bin at the end for >= 1

        hist_precision, binedges = np.histogram(precision, binedges)
        hist_recall, binedges = np.histogram(recall, binedges)

        print("\n\nDistribution of classes with respect to PRECISION: ")
        for b in range(len(binedges) - 1):
            print("[{:.1f}, {:.1f}): {}".format(binedges[b], binedges[b + 1], hist_precision[b]))

        print("\n\nDistribution of classes with respect to RECALL: ")
        for b in range(len(binedges) - 1):
            print("[{:.1f}, {:.1f}): {}".format(binedges[b], binedges[b + 1], hist_recall[b]))

        if self.plot:
            plt.figure()
            plt.subplot(121)
            widths = np.diff(binedges)
            plt.bar(binedges[:-1], hist_precision, width=widths, align='edge')
            plt.xlim(0, 1)
            ax = plt.gca()
            ax.set_xticks(binedges)
            plt.xlabel('Precision')
            plt.ylabel('Number of classes')
            plt.title("Distribution of classes with respect to precision")

            plt.subplot(122)
            widths = np.diff(binedges)
            plt.bar(binedges[:-1], hist_recall, width=widths, align='edge')
            plt.xlim(0, 1)
            ax = plt.gca()
            ax.set_xticks(binedges)
            plt.xlabel('Recall')
            plt.ylabel('Number of classes')
            plt.title("Distribution of classes with respect to recall")

            plt.show(block=False)

    def analyze_classification(self, y_pred, y_true, class_names, excluded_classes=None):
        """
        For an array of label predictions and the respective true labels, shows confusion matrix, accuracy, recall, precision etc:
        Input:
            y_pred: 1D array of predicted labels (class indices)
            y_true: 1D array of true labels (class indices)
            class_names: 1D array or list of class names in the order of class indices.
                Could also be integers [0, 1, ..., num_classes-1].
            excluded_classes: list of classes to be excluded from average precision, recall calculation (e.g. OTHER)
        """

        # Trim class_names to include only classes existing in y_pred OR y_true
        in_pred_labels = set(list(y_pred))
        in_true_labels = set(list(y_true))

        self.existing_class_ind = sorted(list(in_pred_labels | in_true_labels))
        class_strings = [str(name) for name in class_names]  # needed in case `class_names` elements are not strings
        self.existing_class_names = [class_strings[ind][:min(self.maxcharlength, len(class_strings[ind]))] for ind in
                                     self.existing_class_ind]  # a little inefficient but inconsequential

        # Confusion matrix
        ConfMatrix = metrics.confusion_matrix(y_true, y_pred)

        if self.print_conf_mat:
            print_confusion_matrix(ConfMatrix, label_strings=self.existing_class_names, title='Confusion matrix')
            print('\n')
        if self.plot:
            plt.figure()
            plot_confusion_matrix(ConfMatrix, self.existing_class_names)

        # Normalize the confusion matrix by row (i.e by the number of samples in each class)
        self.ConfMatrix_normalized_row = ConfMatrix.astype('float') / ConfMatrix.sum(axis=1)[:, np.newaxis]

        if self.print_conf_mat:
            print_confusion_matrix(self.ConfMatrix_normalized_row, label_strings=self.existing_class_names,
                                   title='Confusion matrix normalized by row')
            print('\n')
        if self.plot:
            plt.figure()
            plot_confusion_matrix(self.ConfMatrix_normalized_row, label_strings=self.existing_class_names,
                                  title='Confusion matrix normalized by row')

            plt.show(block=False)

        # Analyze results
        self.total_accuracy = np.trace(ConfMatrix) / len(y_true)
        # print('Overall accuracy: {:.3f}\n'.format(self.total_accuracy))

        # returns metrics for each class, in the same order as existing_class_names
        self.precision, self.recall, self.f1, self.support = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                                                                     labels=self.existing_class_ind)

        # Print report
        if self.print_conf_mat:
            print(self.generate_classification_report())

        # Calculate average precision and recall
        self.prec_avg, self.rec_avg = self.get_avg_prec_recall(ConfMatrix, self.existing_class_names, excluded_classes)
        if excluded_classes:
            print(
                "\nAverage PRECISION: {:.2f}\n(using class frequencies as weights, excluding classes with no predictions and predictions in '{}')".format(
                    self.prec_avg, ', '.join(excluded_classes)))
            print(
                "\nAverage RECALL (= ACCURACY): {:.2f}\n(using class frequencies as weights, excluding classes in '{}')".format(
                    self.rec_avg, ', '.join(excluded_classes)))

        # Make a histogram with the distribution of classes with respect to precision and recall
        #self.prec_rec_histogram(self.precision, self.recall)

        return {"total_accuracy": self.total_accuracy, "precision": self.precision, "recall": self.recall,
                "f1": self.f1, "support": self.support, "prec_avg": self.prec_avg, "rec_avg": self.rec_avg,
                "ConfMatrix": ConfMatrix}


