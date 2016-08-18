"""
Neural Machine Translation with Reinforcement Bias
"""

from nmt_uni import *
from reward import translation_cost
import time

time = time.time

# utility functions
def _seqs2words(caps, idict):
    capsw = []
    for cc in caps:
        ww = []
        for w in cc:
            if w == 0:
                break
            ww.append(idict[w])
        capsw.append(' '.join(ww))
    return capsw

def _bpe2words(capsw):
    capw = []
    for cc in capsw:
        capw += [cc.replace('@@ ', '')]
    return capw

def _action2delay(src, actions):
    delays = []
    X = len(src)
    for act in actions:
        A = numpy.array(act, dtype='float32')
        Y = numpy.sum(act)
        S = numpy.sum(numpy.cumsum(1 - A) * A)

        assert (X > 0) and (Y > 0), 'avoid NAN {}, {}'.format(X, Y)

        tau = S / (Y * X)
        delays.append([tau, X, Y, S])

    return delays


# padding for computing policy gradient
def _padding(arrays, shape, dtype='float32', return_mask=False, sidx=0):
    B = numpy.zeros(shape, dtype=dtype)

    if return_mask:
        M = numpy.zeros((shape[0], shape[1]), dtype='float32')

    for it, arr in enumerate(arrays):
        arr = numpy.asarray(arr, dtype=dtype)
        # print arr.shape

        steps = arr.shape[0]

        if arr.ndim < 2:
            B[sidx: steps + sidx, it] = arr
        else:
            steps2 = arr.shape[1]
            B[sidx: steps + sidx, it, : steps2] = arr

        if return_mask:
            M[sidx: steps + sidx, it] = 1.

    if return_mask:
        return B, M
    return B


class PIPE(object):
    def __init__(self, keys=None):
        self.messages          = OrderedDict()
        self.hyp_messages      = OrderedDict()
        self.new_hyp_messages  = OrderedDict()
        for key in keys:
            self.messages[key] = []

    def reset(self):
        for key in self.messages:
            self.messages[key] = []

        self.hyp_messages = OrderedDict()
        self.new_hyp_messages = OrderedDict()

    def clean_hyp(self):
        self.hyp_messages      = OrderedDict()

    def clean_new_hyp(self):
        self.new_hyp_messages  = OrderedDict()

    def init_hyp(self, key, live_k=None):
        if live_k is not None:
            self.hyp_messages[key] = [[] for _ in xrange(live_k)]
        else:
            self.hyp_messages[key] = []

    def init_new_hyp(self, key, use_copy=False):
        if use_copy:
            self.new_hyp_messages[key] = copy.copy(self.hyp_messages[key])
        else:
            self.new_hyp_messages[key] = []

    def append(self, key, new, idx=None, use_hyp=False):
        if not use_hyp:
            self.new_hyp_messages[key].append(new)
        else:
            self.new_hyp_messages[key].append(self.hyp_messages[key][idx] + [new])

    def append_new(self, key, idx, hyper=True):
        if hyper:
            self.hyp_messages[key].append(self.new_hyp_messages[key][idx])
        else:
            # print self.messages['sample']
            self.messages[key].append(self.new_hyp_messages[key][idx])

    def add(self, key, new, idx):
        self.new_hyp_messages[key][idx] += new

    def asarray(self, key, replace=False):
        if replace:
            self.hyp_messages[key] = numpy.array(self.hyp_messages[key])
        else:
            return numpy.array(self.hyp_messages[key], dtype='float32')

    def split(self):
        truth  = OrderedDict()
        sample = OrderedDict()


        for key in self.messages:
            if key == 'source':
                continue

            truth[key]  = []
            sample[key] = []

            if key == 'mask':
                for idx in xrange(len(self.messages['source'])):
                    if self.messages['source'][idx] < 0:
                        sample[key].append(self.messages[key][:, idx])
                    else:
                        truth[key].append(self.messages[key][:, idx])
            else:
                for idx in xrange(len(self.messages['source'])):
                    if self.messages['source'][idx] < 0:
                        sample[key].append(self.messages[key][idx])
                    else:
                        truth[key].append(self.messages[key][idx])

        self.messages = sample
        return truth



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
    Statistcs     = OrderedDict()
    n_sentences   = len(srcs)
    max_steps     = -1

    # ======================================================================== #
    # Generating Trajectories based on Current Policy
    # ======================================================================== #

    live_k    = n_samples * n_sentences
    live_all  = live_k

    x, ctx0, z0, secs0 = [], [], [], []
    # data initialization
    for id, (src, trg) in enumerate(zip(srcs, trgs)):

        _x    = numpy.array(src, dtype='int64')[:, None]
        _ctx0 = f_sim_ctx(_x)
        _z0   = f_sim_init(_ctx0[:sidx, :])

        x.append(_x[:, 0])
        ctx0.append(_ctx0[:, 0, :])
        z0.append(_z0.flatten())
        secs0.append([id, len(src), 0])  # word id / source length / correctness

    # pad the results
    x, x_mask = _padding(x, (src_max, n_sentences), dtype='int64', return_mask=True)
    ctx       = _padding(ctx0, (src_max, n_sentences, ctx0[0].shape[-1]))
    z0        = numpy.asarray(z0)
    mask      = x_mask

    # initial actions and hidden states
    action0, _, _, hidden0 = _policy.init_action(n_samples=n_samples)

    x_mask    = numpy.ones_like(x, dtype='float32')
    mask0     = x_mask

    # if we have multiple samples for one input sentence
    mask      = numpy.tile(mask0, [1, n_samples])
    z0        = numpy.tile(z0,    [n_samples, 1])
    ctx       = numpy.tile(ctx,   [1, n_samples, 1])

    hidden0   = numpy.tile(hidden0, [live_k, 1])
    action0   = numpy.tile(action0, [live_k, 1])

    secs      = []
    for _ in xrange(live_k / n_sentences):
        secs += copy.deepcopy(secs0)

    # PIPE for message passing
    pipe      = PIPE(['sample', 'score', 'action', 'obs', 'attentions','secs'])

    # Build for the temporal results: hyp-message
    for key in ['sample', 'obs', 'attentions', 'hidden', 'action']:
        pipe.init_hyp(key, live_k)

    # special care
    pipe.hyp_messages['score']  = numpy.zeros(live_k).astype('float32')
    pipe.hyp_messages['secs']   = secs
    pipe.hyp_messages['states'] = z0
    pipe.hyp_messages['mask']   = mask
    pipe.hyp_messages['ctx']    = ctx

    # these are inputs that needs to be updated
    prev_w     = -1 * numpy.ones((live_k, )).astype('int64')
    prev_z     = z0
    prev_hid   = hidden0
    prev_noise = action0
    step       = 0

    # ROLLOUT: Iteration until all the samples over.
    # Action space:
    # =======================================================================
    while live_k > 0:

        step += 1

        # compute one step
        inps           = [prev_w, ctx, mask, prev_z, prev_noise]
        next_p, _, next_z, next_o, next_a = f_sim_next(*inps)

        # obtain the candidate and the accumulated score.
        _cand          = next_p.argmax(axis=-1)  # live_k
        _score         = next_p[range(live_k), _cand]

        # new place-holders for temporal results: new-hyp-message
        pipe.clean_new_hyp()

        for key in ['sample', 'score', 'attentions', 'secs', 'mask', 'ctx', 'states']:
            pipe.init_new_hyp(key, use_copy=True)

        for key in ['action', 'obs', 'hidden']:
            pipe.init_new_hyp(key, use_copy=False)


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
                    'action', 'obs', 'attentions', 'hidden',
                    'ctx', 'secs', 'mask']:
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

                # *** special care ***
                pipe.hyp_messages['secs'].append(pipe.new_hyp_messages['secs'][idx])
                pipe.hyp_messages['mask'].append(pipe.new_hyp_messages['mask'][:, idx])
                pipe.hyp_messages['ctx'].append(pipe.new_hyp_messages['ctx'][:, idx])



        # make it numpy array
        for key in ['score', 'mask', 'ctx', 'states', 'hidden']:
            pipe.asarray(key, True)

        pipe.hyp_messages['mask'] = pipe.hyp_messages['mask'].T
        if pipe.hyp_messages['ctx'].ndim == 3:
            pipe.hyp_messages['ctx']  = pipe.hyp_messages['ctx'].transpose(1, 0, 2)
        elif pipe.hyp_messages['ctx'].ndim == 2:
            pipe.hyp_messages['ctx']  = pipe.hyp_messages['ctx'][:, None, :]

        prev_z    = pipe.hyp_messages['states']
        prev_hid  = pipe.hyp_messages['hidden']
        mask      = pipe.hyp_messages['mask']
        ctx       = pipe.hyp_messages['ctx']

        prev_w    = numpy.array([w[-1] if len(w) > 0
                                 else -1 for w in pipe.hyp_messages['sample']],
                                dtype='int64')

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

    pipe.messages['R'] = R
    pipe.messages['track'] = track

    # --------------------------------------------------- #
    # add to global lists.
    keywords     = ['sample', 'action', 'obs', 'secs',
                    'attentions', 'score', 'track', 'R']
    for k in keywords:
        if k not in Statistcs:
            Statistcs[k]  = pipe.messages[k]
        else:
            Statistcs[k] += pipe.messages[k]


    # If not train, End here
    if not train:
        return Statistcs

    # ================================================================================================= #
    # Policy Gradient over Trajectories
    # ================================================================================================= #

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


