"""
Neural Machine Translation with Reinforcement Bias
"""

from nmt_uni import *
from simultrans_infinite2 import _seqs2words, _bpe2words, PIPE
from reward import translation_cost

import sys
import time

time = time.time

# ============================================================================ #
# Noisy Decoding in Batch-Mode
# ============================================================================ #
def noisy_decoding(f_sim_ctx,
                   f_sim_init,
                   f_sim_next,
                   f_cost,
                   srcs,     # source sentences
                   trgs,     # taeget sentences
                   t_idict=None,
                   _policy=None,
                   n_samples=10,
                   maxlen=200,
                   reward_config=None,
                   train=False):
    """
    :param f_init:     initializer using the first "sidx" words.
    :param f_sim_next:
    :param f_partial:
    :param src:        the original input needed to be translated (just for the speed)
    :param step:       step_size for each wait
    :param peek:
        hidden0   = _policy.init_hidden()
    :param sidx:       pre-read sidx words from the source
    :return:
    """
    Samples, Actions, Rewards, Observations, Scores, Tracks, Attentions = [], [], [], [], [], [], []
    n_sentences   = len(srcs)
    max_steps     = maxlen

    # ======================================================================== #
    # Generating Trajectories based on Current Policy
    # ======================================================================== #

    for src, trg in zip(srcs, trgs):

        # data initialization
        x         = numpy.array(src, dtype='int64')[:, None]
        ctx0      = f_sim_ctx(x)
        z0        = f_sim_init(ctx0)

        action0, _, _, hidden0 = _policy.init_action(n_samples=n_samples)

        x_mask    = numpy.ones_like(x, dtype='float32')
        mask0     = x_mask

        # if we have multiple samples for one input sentence
        z0        = numpy.tile(z0,   [n_samples, 1])
        ctx       = numpy.tile(ctx0, [1, n_samples, 1])
        mask      = numpy.tile(mask0, [1, n_samples])


        # PIPE for message passing
        pipe      = PIPE(['sample', 'score', 'action', 'obs', 'attentions'])   # action is noise

        # Build for the temporal results: hyp-message
        live_k     = n_samples
        for key in ['sample', 'obs', 'attentions', 'hidden', 'action']:
            pipe.init_hyp(key, live_k)

        # special care
        pipe.hyp_messages['score']    = numpy.zeros(live_k).astype('float32')


        # these are inputs that needs to be updated
        prev_w     = -1 * numpy.ones((live_k, )).astype('int64')
        prev_z     = z0
        prev_hid   = hidden0
        prev_noise = action0
        step       = 0

        # =======================================================================
        # ROLLOUT: Iteration until all the samples over.
        # Action space:
        # =======================================================================
        while live_k > 0:
            # print step, live_k
            # print prev_noise.shape, prev_z.shape, ctx.shape, prev_w.shape, mask.shape

            step += 1
            # compute one step
            inps           = [prev_w, ctx, mask, prev_z, prev_noise]
            next_p, _, next_z, next_o, next_a = f_sim_next(*inps)

            # obtain the candidate and the accumulated score.
            _cand          = next_p.argmax(axis=-1)  # live_k
            _score         = next_p[range(live_k), _cand]

            # new place-holders for temporal results: new-hyp-message
            pipe.clean_new_hyp()

            for key in ['sample', 'score', 'attentions']:
                pipe.init_new_hyp(key, use_copy=True)

            for key in ['action', 'obs', 'hidden']:
                pipe.init_new_hyp(key, use_copy=False)

            # *** special care *** #
            pipe.new_hyp_messages['states'] = copy.copy(prev_z)


            # Rollout the action.
            _actions, _mean, _logstd, _hidden = _policy.action(next_o, prev_hid)  # input the current observation


            # check each candidate
            for idx, wi in enumerate(_cand):

                # collect the action
                a    = _actions[idx]    # 1024-D Gaussian Vector

                # message appending
                pipe.append('obs',       next_o[idx],   idx=idx, use_hyp=True)
                pipe.append('action',    a,             idx=idx, use_hyp=True)   # collect action.
                pipe.append('hidden',    _hidden[idx])

                # for commit:
                # update new_hyp_message
                pipe.add('sample',              [wi], idx)
                pipe.add('score',        _score[idx], idx)
                pipe.add('attentions', [next_a[idx]], idx)

                # *** special care
                pipe.new_hyp_messages['states'][idx]    = next_z[idx]


            #  kill the completed samples, so I need to build new hyp-messages
            pipe.clean_hyp()

            for key in ['sample', 'score', 'states',
                        'action', 'obs', 'attentions', 'hidden' ]:
                pipe.init_hyp(key)


            # print new_hyp_sample
            for idx in xrange(len(pipe.new_hyp_messages['sample'])):
                # check if reachs the end

                if (len(pipe.new_hyp_messages['sample'][idx]) >= maxlen) or \
                        (pipe.new_hyp_messages['sample'][idx][-1] == 0):

                    for key in ['sample', 'score', 'action', 'obs', 'attentions']:
                        pipe.append_new(key, idx, hyper=False)

                    live_k -= 1

                else:

                    for key in ['sample', 'score', 'states', 'action',
                                'obs', 'attentions', 'hidden']:
                        pipe.append_new(key, idx, hyper=True)


            # make it numpy array
            pipe.asarray('score', True)

            prev_z     = pipe.asarray('states')
            prev_hid   = pipe.asarray('hidden')
            # prev_noise = pipe.asarray('action')
            prev_w     = numpy.array([w[-1] if len(w) > 0  else -1 for w in pipe.hyp_messages['sample']], dtype='int64')

            ctx        = numpy.tile(ctx0, [1, live_k, 1])
            mask       = numpy.tile(mask0, [1, live_k])

            prev_noise = numpy.array([a[-1] for a in pipe.hyp_messages['action']], dtype='float32')
            # prev_noise = numpy.concatenate(pipe.hyp_messages['action'], axis=0)


        # =======================================================================
        # Collecting Rewards.
        # =======================================================================
        # print 'collect reward'
        R     = []
        track = []
        reference       = [_bpe2words(_seqs2words([trg], t_idict))[0].split()]
        for k in xrange(n_samples):
            sp, sc, act = [pipe.messages[key][k] for key in ['sample', 'score', 'action']]
            y           = numpy.asarray(sp, dtype='int64')[:, None]
            y_mask      = numpy.ones_like(y, dtype='float32')
            steps       = len(act)

            # turn back to sentence level
            words       = _seqs2words([sp], t_idict)[0]
            decoded     = _bpe2words([words])[0].split()

            # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
            # reward configs
            keys = {"steps": steps, "y": y,
                    "y_mask": y_mask,
                    "x_mask": x_mask,
                    "f_cost": f_cost,
                    "sample": decoded,
                    "reference": reference,
                    "words": words}

            ret  = translation_cost(**keys)
            Rk, bleu = ret

            R     += [Rk]
            track += [bleu]


        # --------------------------------------------------- #
        # add to global lists.

        Samples      += pipe.messages['sample']
        Actions      += pipe.messages['action']
        Observations += pipe.messages['obs']
        Attentions   += pipe.messages['attentions']
        Scores       += pipe.messages['score']
        Tracks       += track
        Rewards      += R


    # If not train, End here
    if not train:
        return Samples, Scores, Actions, Rewards, Tracks, Attentions


    # ================================================================================================= #
    # Policy Gradient over Trajectories
    # ================================================================================================= #

    # padding for computing policy gradient
    def _padding(arrays, shape, dtype='float32', return_mask=False):
        B = numpy.zeros(shape, dtype=dtype)
        if return_mask:
            M = numpy.zeros((shape[0], shape[1]), dtype=dtype)

        for it, arr in enumerate(arrays):
            arr   = numpy.asarray(arr, dtype=dtype)
            steps = arr.shape[0]

            B[: steps, it] = arr
            if return_mask:
                M[: steps, it] = 1.

        if return_mask:
            return B, M
        return B

    # print Act_masks
    # p rint Actions

    # for obs in Observations:
    #     print obs.shape

    p_obs, p_mask   \
            = _padding(Observations,
                       shape=(max_steps, n_samples * n_sentences, _policy.n_in),
                       return_mask=True)
    p_r     = _padding(Rewards,
                       shape=(max_steps, n_samples * n_sentences))
    p_act   = _padding(Actions,
                       shape=(max_steps, n_samples * n_sentences, _policy.n_out))


    # print 'learning policy gradient'
    # learning
    info    = _policy.get_learner()([p_obs, p_mask], p_act, p_r)

    # add the reward statistics
    q = Tracks
    info['Q']   = numpy.mean(q)
    info['A']   = numpy.mean(p_act)

    return Samples, Scores, Actions, Rewards, info


