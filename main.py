import os
from utils import args
from Dataset import dataloader
from models.runner import supervised, pre_training, linear_probing
import numpy as np
import warnings

warnings.simplefilter('ignore')

if __name__ == '__main__':
    config = args.Initialization(args)
    scores, scores_cls = [], []
    for problem in os.listdir(config['data_dir']):
        if problem != 'EEG':
            continue
        config['problem'] = problem
        enable_fine_tuning = False

        while config['k_fold'] >= config['k_fold_cnt']:
            Data = dataloader.data_loader(config)
            if config['Training_mode'] == 'Pre_Training':
                if config['Model_Type'][0] == 'Series2Vec':
                    test_acc, test_cls_acc = pre_training(config, Data, enable_fine_tuning)
                    scores.append(test_acc)
                    scores_cls.append(test_cls_acc.tolist())
            elif config['Training_mode'] == 'Supervised':
                best_aggr_metrics_test, all_metrics = supervised(config, Data)

            config['k_fold_cnt'] += 1
    print(scores)
    print(f'k-fold voting accuracy: {np.mean(scores)}')
    print(scores_cls)
    print(f'k-fold voting accuracy for each class: {np.mean(scores_cls, axis=0)}')


        # print_str = 'Best Model Test Summary: '
        # if best_aggr_metrics_test is not None:
        #     for k, v in best_aggr_metrics_test.items():
        #         print_str += '{}: {} | '.format(k, v)
        #     print(print_str)

        #     with open(os.path.join(config['output_dir'], config['problem']+'_output.txt'), 'w') as file:
        #         for k, v in all_metrics.items():
        #             file.write(f'{k}: {v}\n')
