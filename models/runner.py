import os
from models.model_factory import Model_factory
from models.optimizers import get_optimizer, get_loss_module
from torch.utils.data import DataLoader
from Dataset.dataloader import dataset_class
from models.Series2Vec.S2V_training import *


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix

from utils.utils import load_model


import logging

logger = logging.getLogger('__main__')


def choose_trainer(model, train_loader, test_loader, config, conf_mat, type):
    if config['Model_Type'][0] == 'Series2Vec':
        S_trainer = S2V_S_Trainer(model, train_loader, test_loader, config, print_conf_mat=conf_mat)
    return S_trainer


def rocket_training(config, Data):
    from sktime.classification.kernel_based import RocketClassifier
    clf_name = config['Model_Type'][0]
    logger.info(f"Initializing {clf_name} Classifier ...")

    # Initialize ROCKET Classifier
    model = Model_factory(config, Data)

    # Prepare Data Loaders
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config, meta_info=Data['train_info'])
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config, meta_info=Data['test_info'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    # Convert DataLoader to numpy arrays for RocketClassifier
    def get_data_from_loader(loader):
        data = []
        labels = []
        meta_infos = []

        for i, batch in enumerate(loader):
            X, targets, IDs = batch
            data.append(X.cpu().detach().numpy())
            labels.append(targets.cpu().detach().numpy())
            if loader.dataset.meta_info is not None:
                meta_infos.append(loader.dataset.meta_info[IDs])

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        if loader.dataset.meta_info is not None:
            meta_infos = np.concatenate(meta_infos, axis=0)
            return data, labels, meta_infos
        else:
            return data, labels, None

    train_data, train_labels, train_info = get_data_from_loader(train_loader)
    test_data, test_labels, test_info = get_data_from_loader(test_loader)

    # Train the model
    logger.info(f"Training {clf_name} Classifier ...")
    model.fit(train_data, train_labels)

    # Evaluate the model
    logger.info(f"Evaluating {clf_name} Classifier ...")
    y_hat = model.predict(test_data)
    test_acc, test_class_acc = analysis.subject_wise_analysis(
        y_true=test_labels, 
        y_pred=y_hat, 
        subject_info=test_info,
        epoch_num='pre-training',
        k_fold=config['k_fold_cnt'],
        dataset='test',
        result_path=config['output_dir'])

    return test_acc, test_class_acc


def pre_training(config, Data, enable_fine_tuning=True):
    logger.info("Creating Distance based Self Supervised model ...")
    model = Model_factory(config, Data)
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()
    config['train_info'] = Data['train_info']
    config['test_info'] = Data['test_info']
    model.to(config['device'])

    '''
    the version without fine-tuning
    '''
    # --------------------------------- Load Data ---------------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config, meta_info=Data['train_info'])
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config, meta_info=Data['test_info'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    # --------------------------------- Self Supervised Training ------------------------------------------------------
    SS_trainer = S2V_SS_Trainer(model, train_loader, test_loader, config, print_conf_mat=False)
    save_path = os.path.join(config['save_dir'], config['problem'] + '_model_{}.pth'.format('last'))
    SS_train_runner(config, model, SS_trainer, save_path)

    # --------------------------------------------- Downstream Task (classification)   ---------------------------------
    # ---------------------- Loading the model and freezing layers except FC layer -------------------------------------
    SS_Encoder, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])  # Loading the model
    SS_Encoder.to(config['device'])

    train_repr, train_labels, train_info = S2V_make_representation(SS_Encoder, train_loader)
    test_repr, test_labels, test_info = S2V_make_representation(SS_Encoder, test_loader)
    # clf is ClassiFier
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy(), seed=config['seed'])
    # clf = fit_RidgeClassifier(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    # LP_acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    # print('Test_acc:', LP_acc_test)
    # cm = confusion_matrix(test_labels.cpu().detach().numpy(), y_hat)
    # print("Confusion Matrix:")
    # print(cm)

    test_acc, test_class_acc = analysis.subject_wise_analysis(
        y_true=test_labels.cpu().detach().numpy(), 
        y_pred=y_hat, 
        subject_info=test_info,
        epoch_num='pre-training',
        k_fold=config['k_fold_cnt'],
        dataset='test',
        result_path=config['output_dir'])


    '''
    the version with fine-tuning
    '''
    # --------------------------------- Load Data -------------------------------------------------------------
    if enable_fine_tuning:
        train_dataset = dataset_class(Data['train_data'], Data['train_label'], config, meta_info=Data['train_info'])
        val_dataset = dataset_class(Data['val_data'], Data['val_label'], config, meta_info=Data['val_info'])
        test_dataset = dataset_class(Data['test_data'], Data['test_label'], config, meta_info=Data['test_info'])

        train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

        logger.info('Starting Fine_Tuning...')
        S_trainer = S2V_S_Trainer(SS_Encoder, train_loader, None, config, print_conf_mat=False)
        S_val_evaluator = S2V_S_Trainer(SS_Encoder, val_loader, None, config, print_conf_mat=False)

        save_path = os.path.join(config['save_dir'], config['problem'] + '_2_model_{}.pth'.format('last'))
        S_train_runner(config, SS_Encoder, S_trainer, S_val_evaluator, save_path)

        best_Encoder, optimizer, start_epoch = load_model(SS_Encoder, save_path, config['optimizer'])
        best_Encoder.to(config['device'])

        train_repr, train_labels, train_info = S2V_make_representation(best_Encoder, train_loader)
        test_repr, test_labels, test_info = S2V_make_representation(best_Encoder, test_loader)
        clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
        y_hat = clf.predict(test_repr.cpu().detach().numpy())
        # acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
        # print('Test_acc:', acc_test)
        # cm = confusion_matrix(test_labels.cpu().detach().numpy(), y_hat)
        # print("Confusion Matrix:")
        # print(cm)

        test_acc, test_class_acc = analysis.subject_wise_analysis(y_pred=y_hat, 
                                y_true=test_labels.cpu().detach().numpy(), 
                                subject_info=test_info,
                                epoch_num='fine_tuning',
                                k_fold=config['k_fold_cnt'],
                                dataset='test',
                                result_path=config['output_dir']
                                )

        best_test_evaluator = S2V_S_Trainer(best_Encoder, test_loader, None, config, print_conf_mat=True)
        best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
        # all_metrics['LGR_ACC'] = acc_test #fine-tuning
        # all_metrics['LP_LGR_ACC'] = LP_acc_test #without fine-tuning
        # return best_aggr_metrics_test, all_metrics
    return test_acc, test_class_acc


# This is not implemented by the author
def TS_TCC_pre_training(config, Data):
    logger.info("Creating Distance based Self Supervised model ...")
    model = Model_factory(config, Data)
    temporal_contr_model = TC(config, config['device']).to(config['device'])
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['temp_optimizer'] = optim_class(temporal_contr_model.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()
    model.to(config['device'])

    # --------------------------------- Load Data ---------------------------------------------------------------------
    train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                              collate_fn=lambda x: collate_fn(x))
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                             collate_fn=lambda x: collate_fn(x))
    # --------------------------------- Self Superviseed Training ------------------------------------------------------
    SS_trainer = TS_TCC_SS_Trainer(model, temporal_contr_model, train_loader, test_loader, config, print_conf_mat=False)
    save_path = os.path.join(config['save_dir'], config['problem'] + '_model_{}.pth'.format('last'))
    SS_train_runner(config, model, SS_trainer, save_path)

    # --------------------------------------------- Downstream Task (classification)   ---------------------------------
    # ---------------------- Loading the model and freezing layers except FC layer -------------------------------------
    SS_Encoder, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])  # Loading the model
    SS_Encoder.to(config['device'])
    train_repr, train_labels = TS_TCC_make_representation(SS_Encoder, train_loader)
    test_repr, test_labels = TS_TCC_make_representation(SS_Encoder, test_loader)
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy(), seed=config['seed'])
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    print('Test_acc:', acc_test)
    cm = confusion_matrix(test_labels.cpu().detach().numpy(), y_hat)
    print("Confusion Matrix:")
    print(cm)
    print("F1")
    # print(roc_auc_score(y_hat,test_labels.cpu().detach().numpy()))
    print(f1_score(y_hat, test_labels.cpu().detach().numpy(), average='macro'))

    # --------------------------------- Load Data -------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    logger.info('Starting Fine_Tuning...')
    S_trainer = TS_TCC_S_Trainer(SS_Encoder, None, train_loader, None, config, print_conf_mat=False)
    S_val_evaluator = TS_TCC_S_Trainer(SS_Encoder, None, val_loader, None, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_2_model_{}.pth'.format('last'))
    S_train_runner(config, SS_Encoder, S_trainer, S_val_evaluator, save_path)

    best_Encoder, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
    best_Encoder.to(config['device'])

    best_test_evaluator = TS_TCC_S_Trainer(best_Encoder, None, test_loader, None, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    all_metrics['LGR_ACC'] = acc_test
    all_metrics['LP_LGR_ACC'] = 0
    return best_aggr_metrics_test, all_metrics

# not implemented
def TF_C_pre_training(config, Data):
    logger.info("Creating Distance based Self Supervised model ...")
    model = Model_factory(config, Data)
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()
    model.to(config['device'])

    # --------------------------------- Load Data ---------------------------------------------------------------------
    train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                              collate_fn=lambda x: collate_fn_tfc(x))
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                             collate_fn=lambda x: collate_fn_tfc(x))
    # --------------------------------- Self Superviseed Training ------------------------------------------------------
    SS_trainer = TF_C_SS_Trainer(model, train_loader, test_loader, config, print_conf_mat=False)
    save_path = os.path.join(config['save_dir'], config['problem'] + '_model_{}.pth'.format('last'))
    SS_train_runner(config, model, SS_trainer, save_path)

    # --------------------------------------------- Downstream Task (classification)   ---------------------------------
    # ---------------------- Loading the model and freezing layers except FC layer -------------------------------------
    SS_Encoder, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])  # Loading the model
    SS_Encoder.to(config['device'])
    train_repr, train_labels = make_representation(SS_Encoder, train_loader)
    test_repr, test_labels = make_representation(SS_Encoder, test_loader)
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy(), seed=config['seed'])
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    print('Test_acc:', acc_test)
    cm = confusion_matrix(test_labels.cpu().detach().numpy(), y_hat)
    print("Confusion Matrix:")
    print(cm)
    # --------------------------------- Load Data -------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    logger.info('Starting Fine_Tuning...')
    S_trainer = TF_C_S_Trainer(SS_Encoder, train_loader, None, config, print_conf_mat=False)
    S_val_evaluator = TF_C_S_Trainer(SS_Encoder, val_loader, None, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_2_model_{}.pth'.format('last'))
    S_train_runner(config, SS_Encoder, S_trainer, S_val_evaluator, save_path)

    best_Encoder, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
    best_Encoder.to(config['device'])

    best_test_evaluator = TF_C_S_Trainer(best_Encoder, test_loader, None, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    all_metrics['LGR_ACC'] = acc_test
    all_metrics['LP_LGR_ACC'] = 0
    return best_aggr_metrics_test, all_metrics


def linear_probing():
    return


def supervised(config, Data):
    model = Model_factory(config, Data)
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()
    model.to(config['device'])

    # --------------------------------- Load Data -------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    S_trainer = choose_trainer(model, train_loader, None, config, False, 'S')
    S_val_evaluator = choose_trainer(model, val_loader, None, config, False, 'S')
    save_path = os.path.join(config['save_dir'], config['problem'] + '_model_{}.pth'.format('last'))

    S_train_runner(config, model, S_trainer, S_val_evaluator, save_path)
    best_model, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
    best_model.to(config['device'])

    best_test_evaluator = choose_trainer(best_model, test_loader, None, config, True, 'S')
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    return best_aggr_metrics_test, all_metrics
