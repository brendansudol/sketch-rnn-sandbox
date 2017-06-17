import os

import tensorflow as tf

from magenta.models.sketch_rnn.model import copy_hparams, get_default_hparams
from magenta.models.sketch_rnn.sketch_rnn_train import load_dataset


def load_env(data_dir, model_dir):
    model_params = get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_params.parse_json(f.read())
    return load_dataset(data_dir, model_params, inference_mode=True)


def load_model(model_dir):
    model_params = get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_params.parse_json(f.read())
    model_params.batch_size = 1  # only sample one at a time

    eval_model_params = copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 0

    sample_model_params = copy_hparams(eval_model_params)
    sample_model_params.max_seq_len = 1  # sample one point at a time
    return [model_params, eval_model_params, sample_model_params]
