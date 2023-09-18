import logging
from logging import getLogger
import time
import numpy as np
import pandas as pd
import torch
import sys
from recbole.utils import init_logger, init_seed, set_color
from recbole_gnn.model.social_recommender.diffnet import DiffNet
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer

if __name__ == '__main__':

    config = Config(model='DiffNet', dataset='Ciao', config_file_list=['run_main.yaml'])

    # init random seed
    config['seed'] = 2023
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)
    model =DiffNet(config, train_data.dataset).to(config['device'])
    logger.info(model)
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)

    test_result = trainer.evaluate(test_data)

    print('test_result-[Recall@10,NDCG@10,Precision@10]:', str(test_result['recall@10'])+'\t'+str(test_result['ndcg@10'])+'\t'+str(test_result['precision@10']))
    time.sleep(0.1)
