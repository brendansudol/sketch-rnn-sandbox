import os
import json

import tensorflow as tf

from magenta.models.sketch_rnn.model import get_default_hparams, Model
from magenta.models.sketch_rnn.sketch_rnn_train import (
    FLAGS, load_checkpoint, load_dataset, reset_graph, train
)


DATA_DIR = 'datasets'
MODEL_DIR = 'models/sandwich'
RESUME_TRAINING = False
OPTIONS = {
    'augment_stroke_prob': 0.15,
    'kl_decay_rate': 0.9999,
    'kl_weight': 1.0,
    'data_set': ['sandwich.npz'],
    'max_seq_len': 174,
    'use_recurrent_dropout': 0,
}

# TODO: rework this bit
FLAGS.log_root = MODEL_DIR
FLAGS.resume_training = RESUME_TRAINING


def run(params):
    tf.logging.info('hyperparams...')
    for key, val in params.values().iteritems():
        tf.logging.info('%s = %s', key, str(val))

    tf.logging.info('loading data files...')
    datasets = load_dataset(DATA_DIR, params)
    [train_set, valid_set, test_set, model_params, eval_model_params, _] = datasets

    reset_graph()
    model = Model(model_params)
    eval_model = Model(eval_model_params, reuse=True)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    if RESUME_TRAINING:
        tf.logging.info('resume training...')
        load_checkpoint(sess, MODEL_DIR)

    tf.gfile.MakeDirs(MODEL_DIR)
    with tf.gfile.Open(os.path.join(MODEL_DIR, 'model_config.json'), 'w') as f:
        json.dump(model_params.values(), f, indent=True)

    train(sess, model, eval_model, train_set, valid_set, test_set)


params = get_default_hparams()
params.parse_json(json.dumps(OPTIONS))
run(params)
