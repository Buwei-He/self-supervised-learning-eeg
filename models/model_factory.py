import logging
from models.Series2Vec import Series2Vec
from sktime.classification.kernel_based import RocketClassifier

####
logger = logging.getLogger('__main__')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def Model_factory(config, data):
    config['Data_shape'] = data['train_data'].shape
    config['num_labels'] = int(max(data['train_label'])) + 1

    if config['Model_Type'][0] == 'Series2Vec':
        model = Series2Vec.Seires2Vec(config, num_classes=config['num_labels'])
        logger.info("Total number of parameters: {}".format(count_parameters(model)))
    if config['Model_Type'][0] == 'TS_TCC':
        model = TS_TCC.TS_TCC(config, num_classes=config['num_labels'])
    if config['Model_Type'][0] == 'TF_C':
        model = TF_C.TF_C(config, num_classes=config['num_labels'])
    if config['Model_Type'][0] in ['rocket', 'minirocket', 'multirocket']:
        # “rocket”, “minirocket”, “multirocket”
        model = RocketClassifier(rocket_transform=config['Model_Type'][0], n_jobs=-1)

    logger.info("Model:\n{}".format(model))
    return model
