"""Script for training translation models and decoding from them

See the following papers for more information on neural translation models
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
import os
import sys
import logging
import argparse
import subprocess
import tensorflow as tf
import yaml
import shutil

from pprint import pformat
from operator import itemgetter
from translate import utils
from translate.translation_model import TranslationModel

parser = argparse.ArgumentParser()
parser.add_argument('config', help='load a configuration file in the YAML format')
parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')
parser.add_argument('--reset', help="reset model (don't load any checkpoint)", action='store_true')
parser.add_argument('--purge', help='remove previous model files', action='store_true')

# Available actions (exclusive)
parser.add_argument('--decode', help='translate this corpus (one filename for each encoder)', nargs='*')
parser.add_argument('--eval', help='compute BLEU score on this corpus (source files and target file)', nargs='+')
parser.add_argument('--train', help='train an NMT model', action='store_true')

# TensorFlow configuration
parser.add_argument('--gpu-id', type=int, help='index of the GPU where to run the computation')
parser.add_argument('--no-gpu', action='store_true', help='run on CPU')

# Decoding options (to avoid having to edit the config file)
parser.add_argument('--beam-size', type=int)
parser.add_argument('--checkpoints', nargs='+')
parser.add_argument('--output')
parser.add_argument('--max-steps', type=int)
parser.add_argument('--remove-unk', action='store_const', const=True)


def main(args=None):
    args = parser.parse_args(args)

    # read config file and default config
    with open('config/default.yaml') as f:
        default_config = utils.AttrDict(yaml.safe_load(f))

    with open(args.config) as f:
        config = utils.AttrDict(yaml.safe_load(f))
        # command-line parameters have higher precedence than config file
        for k, v in vars(args).items():
            if v is not None and (k in default_config or k in ('decode', 'eval', 'output')):
                config[k] = v

        # set default values for parameters that are not defined
        for k, v in default_config.items():
            config.setdefault(k, v)

    # enforce parameter constraints
    assert config.steps_per_eval % config.steps_per_checkpoint == 0, (
        'steps-per-eval should be a multiple of steps-per-checkpoint')
    assert args.decode is not None or args.eval or args.train, (
        'you need to specify at least one action (decode, eval, or train)')

    if args.purge:
        utils.log('deleting previous model')
        shutil.rmtree(config.model_dir, ignore_errors=True)

    logging_level = logging.DEBUG if args.verbose else logging.INFO
    # always log to stdout in decoding and eval modes (to avoid overwriting precious train logs)
    logger = utils.create_logger(config.log_file if args.train else None)
    logger.setLevel(logging_level)

    utils.log(' '.join(sys.argv))  # print command line
    try:  # print git hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        utils.log('commit hash {}'.format(commit_hash))
    except:
        pass

    # list of encoder and decoder parameter names (each encoder and decoder can have a different value
    # for those parameters)
    model_parameters = ['cell_size', 'layers', 'vocab_size', 'embedding_size', 'use_lstm', 'bidir', 'swap_memory',
        'parallel_iterations', 'attn_size']

    config.encoder = utils.AttrDict(config.encoder)
    config.decoder = utils.AttrDict(config.decoder)

    for parameter in model_parameters:
        if parameter not in config.encoder:
            config.encoder[parameter] = config.get(parameter)
        if parameter not in config.decoder:
            config.decoder[parameter] = config.get(parameter)

    # log parameters
    utils.log('program arguments')
    for k, v in sorted(config.items(), key=itemgetter(0)):
        utils.log('  {:<20} {}'.format(k, pformat(v)))

    device = None
    if args.no_gpu:
        device = '/cpu:0'
    elif args.gpu_id is not None:
        device = '/gpu:{}'.format(args.gpu_id)

    utils.log('creating model')
    utils.log('using device: {}'.format(device))

    with tf.device(device):
        checkpoint_dir = os.path.join(config.model_dir, 'checkpoints')
        # All parameters except recurrent connexions and attention parameters are initialized with this.
        # Recurrent connexions are initialized with orthogonal matrices, and the parameters of the attention model
        # with a standard deviation of 0.001
        if config.weight_scale:
            initializer = tf.random_normal_initializer(stddev=config.weight_scale)
        else:
            initializer = None

        tf.get_variable_scope().set_initializer(initializer)
        decode_only = args.decode is not None or args.eval  # exempt from creating gradient ops
        model = TranslationModel(name='main', checkpoint_dir=checkpoint_dir, decode_only=decode_only, **config)

    utils.log('model parameters ({})'.format(len(tf.all_variables())))
    parameter_count = 0
    for var in tf.all_variables():
        utils.log('  {} {}'.format(var.name, var.get_shape()))

        v = 1
        for d in var.get_shape():
            v *= d.value
        parameter_count += v
    utils.log('number of parameters: {}'.format(parameter_count))

    tf_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = config.allow_growth
    tf_config.gpu_options.per_process_gpu_memory_fraction = config.mem_fraction

    with tf.Session(config=tf_config) as sess:
        best_checkpoint = os.path.join(checkpoint_dir, 'best')

        if (not config.checkpoints and not args.reset and (args.eval or args.decode is not None)
              and os.path.isfile(best_checkpoint)):
            # in decoding and evaluation mode, unless specified otherwise (by `checkpoints` or `reset` parameters,
            # try to load the best checkpoint)
            model.initialize(sess, [best_checkpoint], reset=True)
        else:
            # loads last checkpoint, unless `reset` is true
            model.initialize(sess, config.checkpoints, reset=args.reset)

        # Inspect variables:
        # tf.get_variable_scope().reuse_variables()
        # import pdb; pdb.set_trace()

        if args.decode is not None:
            model.decode(sess, **config)
        elif args.eval:
            model.evaluate(sess, on_dev=False, **config)
        elif args.train:
            eval_output = os.path.join(config.model_dir, 'eval')
            try:
                model.train(sess, eval_output=eval_output, **config)
            except KeyboardInterrupt:
                utils.log('exiting...')
                model.save(sess)
                sys.exit()


if __name__ == '__main__':
    main()
