"""
Neural Machine Translation with Greedy Decoding
"""
import argparse
import os
import cPickle as pkl

from nmt_uni import *
from policy import Controller as Policy
from utils import Progbar, Monitor
from noisy_translator import noisy_decoding
from simultrans_infinite2 import _seqs2words, _bpe2words, _action2delay

import time


numpy.random.seed(19920206)
timer = time.time

# check hidden folders
def check_env():
    paths = ['.policy', '.pretrained', '.log',
             '.config', '.images', '.translate']
    for p in paths:
        if not os.path.exists(p):
            os.mkdir(p)


# run training function:: >>>
def run_simultrans(model,
                   options_file=None,
                   config=None,
                   policy=None,
                   id=None,
                   remote=False):
    # check envoriments
    check_env()
    if id is not None:
        fcon = '.config/{}.conf'.format(id)
        if os.path.exists(fcon):
            print 'load config files'
            policy, config = pkl.load(open(fcon, 'r'))

    # ======================================================================= #
    # load model model_options
    # ======================================================================= #
    _model = model
    model  = '.pretrained/{}'.format(model)

    if options_file is not None:
        with open(options_file, 'rb') as f:
            options = pkl.load(f)
    else:
        with open('%s.pkl' % model, 'rb') as f:
            options = pkl.load(f)
    options['birnn'] = True

    print 'load options...'
    for w, p in sorted(options.items(), key=lambda x:x[0]):
        print '{}: {}'.format(w, p)

    # load detail settings from option file:
    dictionary, dictionary_target = options['dictionaries']

    def _iter(fname):
        with open(fname, 'r') as f:
            for line in f:
                words = line.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words'] else 1, x)
                x += [0]
                yield x

    def _check_length(fname):
        f = open(fname, 'r')
        count = 0
        for _ in f:
            count += 1
        f.close()

        return count

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # ======================================================================== #
    # Build a Translator
    # ======================================================================== #

    # allocate model parameters
    params  = init_params(options)
    params  = load_params(model, params)
    tparams = init_tparams(params)

    # print 'build the model for computing cost (full source sentence).'
    trng, use_noise, \
    _x, _x_mask, _y, _y_mask, \
    opt_ret, \
    cost, f_cost = build_model(tparams, options)
    print 'done.'

    # functions for sampler
    # f_sim_ctx, f_sim_init, f_sim_next = build_simultaneous_sampler(tparams, options, trng)
    f_sim_ctx, f_sim_init, f_sim_next = build_noisy_sampler(tparams, options, trng)
    print 'build sampler done.'

    # check the ID:
    policy['base'] = _model
    _policy        = Policy(trng, options, policy, config,
                            n_out=options['dim'],
                            recurrent=True, id=id)


    # DATASET
    trainIter = TextIterator(options['datasets'][0], options['datasets'][1],
                             options['dictionaries'][0], options['dictionaries'][1],
                             n_words_source=options['n_words_src'], n_words_target=options['n_words'],
                             batch_size=config['batchsize'],
                             maxlen=options['maxlen'])

    train_num = trainIter.num

    validIter = TextIterator(options['valid_datasets'][0], options['valid_datasets'][1],
                             options['dictionaries'][0], options['dictionaries'][1],
                             n_words_source=options['n_words_src'], n_words_target=options['n_words'],
                             batch_size=1,
                             maxlen=options['maxlen'])

    valid_num = validIter.num

    valid_    = options['valid_datasets'][0]
    valid_num = _check_length(valid_)
    print 'training set {} lines / validation set {} lines'.format(train_num, valid_num)
    print 'use the reward function {}'.format(chr(config['Rtype'] + 65))

    # Translator model
    def _translate(src, trg, train=False, samples=80):
        ret = noisy_decoding(
            f_sim_ctx, f_sim_init,
            f_sim_next, f_cost,
            src, trg, word_idict_trg, n_samples=samples,
            train=train,
            _policy=_policy)

        if not train:
            sample, score, actions, R, tracks, attentions = ret
            return sample, score, actions, R, tracks
        else:
            sample, score, actions, R, info = ret
            return sample, score, actions, R, info


    # ======================================================================== #
    # Main Loop: Run
    # ======================================================================== #
    print 'Start Simultaneous Translator...'
    probar           = Progbar(train_num / config['batchsize'],  with_history=False)

    # freqs
    save_freq        = 2000
    sample_freq      = 10
    valid_freq       = 1000
    valid_size       = 200
    display_freq     = 50

    history, last_it = _policy.load()
    time0            = timer()

    for it, (srcs, trgs) in enumerate(trainIter):  # only one sentence each iteration
        if it < last_it:  # go over the scanned lines.
            continue

        samples, scores, actions, rewards, info = _translate(srcs, trgs, train=True)
        if it % sample_freq == 0:

            print '\nModel has been trained for {} seconds'.format(timer() - time0)
            print 'source: ', _bpe2words(_seqs2words([srcs[0]], word_idict))[0]
            print 'target: ', _bpe2words(_seqs2words([trgs[0]], word_idict_trg))[0]

            # obtain the translation results
            samples = _bpe2words(_seqs2words(samples, word_idict_trg))

            print '---'
            print 'sample: ', samples[40]
            print 'sample: ', samples[60]

        values = [(w, info[w]) for w in info]
        probar.update(it + 1, values=values)

        # NaN detector
        for w in info:
            if numpy.isnan(info[w]) or numpy.isinf(info[w]):
                raise RuntimeError, 'NaN/INF is detected!! {} : ID={}'.format(w, id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        default='model_wmt15_bpe2k_basic_cs-en.npz')
    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('-o', type=str, default=None)

    args   = parser.parse_args()
    print args

    policy = OrderedDict()
    policy['layernorm'] = True
    policy['upper']     = False
    policy['updater']   = 'REINFORCE'
    policy['type']      = 'gaussian'

    config = OrderedDict()
    config['batchsize'] = 1
    config['Rtype']     = 8

    run_simultrans(args.model,
                   options_file=args.o,
                   config=config,
                   policy=policy,
                   id=args.id,
                   remote=False)


