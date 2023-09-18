import logging
import time
from logging import getLogger
from recbole.utils import init_logger, init_seed, set_color
from recbole_gnn.model.social_recommender.cirec import CIRec
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer
import sys

if __name__ == '__main__':

    config = Config(model='CIRec', dataset='Douban', config_file_list=['run_counterfactual_generation.yaml'])
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    start_time1 = time.time()
    model = CIRec(config, train_data.dataset).to(config['device'])
    stop_time1 = time.time()
    print('Time spent generating counterfactual data：', stop_time1-start_time1, 's')