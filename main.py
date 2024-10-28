import os
from utils import args
from Dataset import dataloader
from models.runner import supervised, pre_training, linear_probing


if __name__ == '__main__':
    config = args.Initialization(args)

for problem in os.listdir(config['data_dir']):
    if problem != 'EEG':
        continue
    config['problem'] = problem
    enable_fine_tuning = True
    Data = dataloader.data_loader(config)
    best_aggr_metrics_test = None
    all_metrics = None

    if config['Training_mode'] == 'Pre_Training':
        if config['Model_Type'][0] == 'Series2Vec':
            best_aggr_metrics_test, all_metrics = pre_training(config, Data, enable_fine_tuning)
    elif config['Training_mode'] == 'Linear_Probing':
        best_aggr_metrics_test, all_metrics = linear_probing(config, Data)
    elif config['Training_mode'] == 'Supervised':
        best_aggr_metrics_test, all_metrics = supervised(config, Data)

    print_str = 'Best Model Test Summary: '
    if best_aggr_metrics_test is not None:
        for k, v in best_aggr_metrics_test.items():
            print_str += '{}: {} | '.format(k, v)
        print(print_str)

        with open(os.path.join(config['output_dir'], config['problem']+'_output.txt'), 'w') as file:
            for k, v in all_metrics.items():
                file.write(f'{k}: {v}\n')
