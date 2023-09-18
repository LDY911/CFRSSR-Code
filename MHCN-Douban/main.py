import os
import random
import numpy as np
import tensorflow as tf
seed_value = 2023
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

from SELFRec import SELFRec
from util.conf import ModelConf


if __name__ == '__main__':
    graph_baselines = ['DirectAU', 'MF', 'SASRec']
    ssl_graph_models = ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL', 'MixGCF']
    sequential_baselines = ['SASRec']
    ssl_sequential_models = ['CL4SRec', 'DuoRec', 'BERT4Rec']

    model = 'MHCN'
    import time

    s = time.time()
    if model in graph_baselines or model in ssl_graph_models or model in sequential_baselines or model in ssl_sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')

    else:
        print('Wrong model name!')
        exit(-1)

    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
    time.sleep(5)
