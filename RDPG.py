"""
-- Recurrent Deterministic Policy Gradient
"""

from nmt_uni import *

import os
import time, datetime
import cPickle as pkl


class RDPG(object):

    def __init__(self,
                 trng, options, policy, config,
                 n_in=None, n_out=None,
                 recurrent=False, id=None):

        self.trng      = trng
        self.options   = options
        self.policy    = policy
        self.recurrent = recurrent

        self.n_hidden  = 512
        self.n_in      = n_in
        self.n_out     = n_out

        self.rec       = 'lngru'
        if not n_in:
            self.n_in  = options['readout_dim']

        # ------------------------------------------------------------------------------
        print 'policy network initialization'

        params = OrderedDict()
        if not self.recurrent:
            print 'building a feed-forward controller'
            params = get_layer('ff')[0](options, params, prefix='policy_net_in',
                                        nin=self.n_in, nout=self.n_hidden, scale=0.001)
        else:
            print 'building a recurrent controller'
            params = get_layer(self.rec)[0](options, params, prefix='policy_net_in',
                                            nin=self.n_in, dim=self.n_hidden, scale=0.001)

        params = get_layer('ff')[0](options, params, prefix='policy_net_out',
                                    nin=self.n_hidden,
                                    nout=self.n_out,
                                    scale=0.001)

        # --------------------------------------------------------------------------------
        print 'critic network initialization (RNN)'
        params_b = OrderedDict()
        params_b = get_layer(self.rec)[0](options, params_b, prefix='critic_net_in',
                                            nin=self.n_in + self.n_out,
                                          dim=self.n_hidden, scale=0.001)
        params_b = get_layer('ff')[0](options, params_b, prefix='critic_net_out',
                                      nin=self.n_hidden,
                                      nout=1,
                                      scale=0.001)
        if id is not None:
            print 'reload the saved model: {}'.format(id)
            params = load_params('.policy/{}-{}.current.npz'.format(id, self.policy['base']), params)
            params_b = load_params('.policy/{}-{}.current.npz'.format(id, self.policy['base']), params_b)
        else:
            id = datetime.datetime.fromtimestamp(time.time()).strftime('%y%m%d-%H%M%S')
            print 'start from a new model: {}'.format(id)

            with open('.config/conf.{}.txt'.format(id), 'w') as f:
                f.write('[config]\n')

                for c in config:
                    f.write('{}: {}\n'.format(c, config[c]))
                f.write('\n')

                f.write('[policy]\n')

                for c in policy:
                    f.write('{}: {}\n'.format(c, policy[c]))

            # pkl.dump([policy, config], open('.config/{}.conf'.format(id), 'w'))
            print 'save the config file'

        self.id = id
        self.model = '.policy/{}-{}'.format(id, self.policy['base'])

        # theano shared params
        self.tparams   = init_tparams(params)
        self.tparams_b = init_tparams(params_b)

        # build the policy network
        self.build_actor(options=options)
        self.build_discriminator(options=options)

    def build_actor(self, options):
        # ============================================================================= #
        # Actor from Policy Network
        # ============================================================================= #
        observation = tensor.matrix('observation', dtype='float32')  # batch_size x readout_dim (seq_steps=1)
        prev_hidden = tensor.matrix('p_hidden', dtype='float32')

        if not self.recurrent:
            hiddens = get_layer('ff')[1](self.tparams, observation,
                                         options, prefix='policy_net_in',
                                         activ='tanh')
        else:
            hiddens = get_layer(self.rec)[1](self.tparams, observation,
                                             options, prefix='policy_net_in', mask=None,
                                             one_step=True, _init_state=prev_hidden)[0]

        act_inps = [observation, prev_hidden]
        act_outs = get_layer('ff')[1](self.tparams, hiddens, options,
                                      prefix='policy_net_out',
                                      activ='tanh'
                                      )
        print 'build action function [Deterministic]'
        self.f_action = theano.function(act_inps, act_outs,
                                        on_unused_input='ignore')  # action/dist/hiddens
        print 'done.'


    def build_discriminator(self, options):
        # ============================================================================= #
        # Build for End-t-End learning
        # ============================================================================= #
        observations = tensor.tensor3('observations', dtype='float32')
        mask         = tensor.matrix('mask', dtype='float32')
        targets      = tensor.vector('targets', dtype='float32')

        print 'build actor'
        if not self.recurrent:
            hiddens  = get_layer('ff')[1](self.tparams, observations,
                                         options, prefix='policy_net_in',
                                         activ='tanh')
        else:
            hiddens  = get_layer(self.rec)[1](self.tparams, observations,
                                             options, prefix='policy_net_in', mask=mask)[0]
        actions      = get_layer('ff')[1](self.tparams, hiddens, options, prefix='policy_net_out',
                                          activ='tanh')  # seq_steps x batch_size x n_out

        print 'build critic'
        state_action = concatenate([observations, actions], axis=-1)
        hiddens_b    = get_layer(self.rec)[1](self.tparams_b, state_action,
                                             options, prefix='critic_net_in', mask=mask)[0]
        values       = get_layer('ff')[1](self.tparams_b, hiddens_b, options,
                                           prefix='critic_net_out',
                                           activ='tanh')[-1, :, 0]  # (batch_size, )

        # =============================================================================== #
        # Build Deterministic Policy Gradient [Actor Parts]
        # =============================================================================== #
        inps_A       = [observations, mask]
        loss_A       = -tensor.mean(values)
        grad_A       = tensor.grad(loss_A, wrt=itemlist(self.tparams))
        grad_A       = grad_clip(grad_A)
        outs_A       = [loss_A, actions]

        # optimizer: Adam
        lr           = tensor.scalar(name='lr')
        f_A, f_Aup   = adam(lr, self.tparams, grad_A, inps_A, outs_A)

        # =============================================================================== #
        # Build Deterministic Policy Gradient [Critic Parts]
        # =============================================================================== #
        inps_B       = [observations, mask, actions, targets]
        loss_B       = tensor.mean((values - targets) ** 2)
        grad_B       = tensor.grad(loss_B, wrt=itemlist(self.tparams_b))
        grad_B       = grad_clip(grad_B)
        outs_B       = [loss_B]

        # optimizer: Adam
        lr           = tensor.scalar(name='lr')
        f_B, f_Bup   = adam(lr, self.tparams_b, grad_B, inps_B, outs_B)

        self.f_learner = [f_A, f_Aup, f_B, f_Bup]
        print 'done.'



