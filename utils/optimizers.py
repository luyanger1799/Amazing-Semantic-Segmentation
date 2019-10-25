"""
The implementation of some optimizers based on Tensorflow.

@Author: OverLordGoldDragon
@Project: https://github.com/OverLordGoldDragon/keras-adamw

"""
from termcolor import colored
import tensorflow as tf
import numpy as np

K = tf.keras.backend
Optimizer = tf.keras.optimizers.Optimizer


def warn_str():
    return colored('WARNING: ', 'red')


class AdamW(Optimizer):
    """AdamW optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond".
        batch_size:       int >= 1. Train input batch size; used for normalization
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [2]-Appendix, 2
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [2]-Appendix, 2
        t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
               To be used together with use_cosine_annealing==True
        init_verbose: bool. If True, print weight-name--weight-decay, and
                      lr-multiplier--layer-name value pairs set during
                      optimizer initialization (recommended)
    # <1> - if using 'warm restarts', then refers to total expected iterations
            for a given restart; can be an estimate, and training won't stop
            at iterations == total_iterations. [2]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
            (https://github.com/OverLordGoldDragon/keras_adamw)
    # References
        - [1][Adam - A Method for Stochastic Optimization]
             (http://arxiv.org/abs/1412.6980v8)
        - [2][Fixing Weight Decay Regularization in Adam]
             (https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 amsgrad=False, batch_size=32, total_iterations=0,
                 weight_decays=None, lr_multipliers=None,
                 use_cosine_annealing=False, eta_min=0, eta_max=1,
                 t_cur=0, init_verbose=True, name='AdamW', **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        eta_t = kwargs.pop('eta_t', 1.)
        super(AdamW, self).__init__(name, **kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(self.initial_decay, name='decay')
            self.batch_size = K.variable(batch_size, dtype='int64',
                                         name='batch_size')
            self.total_iterations = K.variable(total_iterations, dtype='int64',
                                               name='total_iterations')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        self.amsgrad = amsgrad
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.updates.append(K.update_add(self.t_cur, 1))

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p),
                      dtype=K.dtype(p),
                      name='m_' + str(i))
              for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p),
                      dtype=K.dtype(p),
                      name='v_' + str(i))
              for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p),
                             dtype=K.dtype(p),
                             name='vhat_' + str(i))
                     for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i))
                     for i in range(len(params))]
        self.weights = [self.iterations] + ms + vs + vhats

        total_iterations = K.get_value(self.total_iterations)
        if total_iterations == 0:
            print(warn_str() + "'total_iterations'==0, must be !=0 to use "
                  + "cosine annealing and/or weight decays; "
                  + "proceeding without either")
        # Schedule multiplier
        if self.use_cosine_annealing and total_iterations != 0:
            t_frac = K.cast(self.t_cur / (self.total_iterations + 1), 'float32')
            self.eta_t = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
                         (1 + K.cos(np.pi * t_frac))
            if self.init_verbose:
                print('Using cosine annealing learning rates')
        self.lr_t = lr * self.eta_t  # for external tracking

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # Learning rate multipliers
            multiplier_name = None
            if self.lr_multipliers:
                multiplier_name = [mult_name for mult_name in self.lr_multipliers
                                   if mult_name in p.name]
            new_lr = lr_t
            if multiplier_name:
                new_lr = new_lr * self.lr_multipliers[multiplier_name[0]]
                if self.init_verbose:
                    print('{} learning rate set for {} -- {}'.format(
                        '%.e' % K.get_value(new_lr), p.name.split('/')[0], new_lr))
            elif not multiplier_name and self.init_verbose:
                print('No change in learning rate {} -- {}'.format(
                    p.name, K.get_value(new_lr)))

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            # Weight decays
            if p.name in self.weight_decays.keys() and total_iterations != 0:
                wd = self.weight_decays[p.name]
                wd_normalized = wd * K.cast(
                    K.sqrt(self.batch_size / self.total_iterations), 'float32')
                p_t = p_t - self.eta_t * wd_normalized * p
                if self.init_verbose:
                    print('{} weight decay set for {}'.format(
                        K.get_value(wd_normalized), p.name))

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'batch_size': int(K.get_value(self.batch_size)),
                  'total_iterations': int(K.get_value(self.total_iterations)),
                  'weight_decays': self.weight_decays,
                  'lr_multipliers': self.lr_multipliers,
                  'use_cosine_annealing': self.use_cosine_annealing,
                  't_cur': int(K.get_value(self.t_cur)),
                  'eta_t': int(K.get_value(self.eta_t)),
                  'eta_min': int(K.get_value(self.eta_min)),
                  'eta_max': int(K.get_value(self.eta_max)),
                  'init_verbose': self.init_verbose,
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NadamW(Optimizer):
    """Nesterov Adam optimizer.
    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.
    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
    # Arguments (other): see AdamW
    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning]
          (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """

    def __init__(self, learning_rate=0.002, beta_1=0.9, beta_2=0.999,
                 batch_size=32, total_iterations=0,
                 weight_decays=None, lr_multipliers=None,
                 use_cosine_annealing=False, eta_min=0, eta_max=1,
                 t_cur=0, init_verbose=True, name='NadamW', **kwargs):
        self.schedule_decay = kwargs.pop('schedule_decay', 0.004)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        eta_t = kwargs.pop('eta_t', 1.)
        super(NadamW, self).__init__(name, **kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.batch_size = K.variable(batch_size, dtype='int64',
                                         name='batch_size')
            self.total_iterations = K.variable(total_iterations, dtype='int64',
                                               name='total_iterations')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays
        self.use_cosine_annealing = use_cosine_annealing
        self.init_verbose = init_verbose

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.updates.append(K.update_add(self.t_cur, 1))

        t = K.cast(self.iterations, K.floatx()) + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape, name='m_' + str(i))
              for (i, shape) in enumerate(shapes)]
        vs = [K.zeros(shape, name='v_' + str(i))
              for (i, shape) in enumerate(shapes)]

        self.weights = [self.iterations, self.m_schedule] + ms + vs

        total_iterations = K.get_value(self.total_iterations)
        if total_iterations == 0:
            print(warn_str() + "'total_iterations'==0, must be !=0 to use "
                  + "cosine annealing and/or weight decays; "
                  + "proceeding without either")
        # Schedule multiplier
        if self.use_cosine_annealing and total_iterations != 0:
            t_frac = K.cast(self.t_cur / (self.total_iterations + 1), 'float32')
            self.eta_t = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
                         (1 + K.cos(np.pi * t_frac))
            if self.init_verbose:
                print('Using cosine annealing learning rates')
        self.lr_t = self.learning_rate * self.eta_t  # for external tracking

        for p, g, m, v in zip(params, grads, ms, vs):
            # Learning rate multipliers
            multiplier_name = None
            if self.lr_multipliers:
                multiplier_name = [mult_name for mult_name in self.lr_multipliers
                                   if mult_name in p.name]
            new_lr = self.learning_rate
            if multiplier_name:
                new_lr = new_lr * self.lr_multipliers[multiplier_name[0]]
                if self.init_verbose:
                    print('{} learning rate set for {} -- {}'.format(
                        '%.e' % K.get_value(new_lr), p.name.split('/')[0], new_lr))
            elif not multiplier_name and self.init_verbose:
                print('No change in learning rate {} -- {}'.format(
                    p.name, K.get_value(new_lr)))

            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + (
                    momentum_cache_t_1 * m_t_prime)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            p_t = p - self.eta_t * new_lr * m_t_bar / (
                    K.sqrt(v_t_prime) + self.epsilon)

            # Weight decays
            if p.name in self.weight_decays.keys() and total_iterations != 0:
                wd = self.weight_decays[p.name]
                wd_normalized = wd * K.cast(
                    K.sqrt(self.batch_size / self.total_iterations), 'float32')
                p_t = p_t - self.eta_t * wd_normalized * p
                if self.init_verbose:
                    print('{} weight decay set for {}'.format(
                        K.get_value(wd_normalized), p.name))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras 2.2.4 optimizer
        # since it does not include m_schedule at head of the weight list. Set
        # m_schedule to 1.
        if len(params) == len(weights) + 1:
            weights = [weights[0]] + [np.array(1.)] + weights[1:]
        super(NadamW, self).set_weights(weights)

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon,
                  'schedule_decay': self.schedule_decay,
                  'batch_size': int(K.get_value(self.batch_size)),
                  'total_iterations': int(K.get_value(self.total_iterations)),
                  'weight_decays': self.weight_decays,
                  'lr_multipliers': self.lr_multipliers,
                  'use_cosine_annealing': self.use_cosine_annealing,
                  't_cur': int(K.get_value(self.t_cur)),
                  'eta_t': int(K.get_value(self.eta_t)),
                  'eta_min': int(K.get_value(self.eta_min)),
                  'eta_max': int(K.get_value(self.eta_max)),
                  'init_verbose': self.init_verbose}
        base_config = super(NadamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SGDW(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    # Arguments (other): see AdamW
    """

    def __init__(self, learning_rate=0.01, momentum=0., nesterov=False,
                 batch_size=32, total_iterations=0,
                 weight_decays=None, lr_multipliers=None,
                 use_cosine_annealing=False, eta_min=0, eta_max=1,
                 t_cur=0, init_verbose=True, name='SGDW', **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        learning_rate = kwargs.pop('lr', learning_rate)
        eta_t = kwargs.pop('eta_t', 1.)
        super(SGDW, self).__init__(name, **kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(self.initial_decay, name='decay')
            self.batch_size = K.variable(batch_size, dtype='int64',
                                         name='batch_size')
            self.total_iterations = K.variable(total_iterations, dtype='int64',
                                               name='total_iterations')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        self.nesterov = nesterov
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.updates.append(K.update_add(self.t_cur, 1))

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape, name='moment_' + str(i))
                   for (i, shape) in enumerate(shapes)]
        self.weights = [self.iterations] + moments

        total_iterations = K.get_value(self.total_iterations)
        if total_iterations == 0:
            print(warn_str() + "'total_iterations'==0, must be !=0 to use "
                  + "cosine annealing and/or weight decays; "
                  + "proceeding without either")
        # Schedule multiplier
        if self.use_cosine_annealing and total_iterations != 0:
            t_frac = K.cast(self.t_cur / (self.total_iterations + 1), 'float32')
            self.eta_t = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
                         (1 + K.cos(np.pi * t_frac))
            if self.init_verbose:
                print('Using cosine annealing learning rates')
        self.lr_t = lr * self.eta_t  # for external tracking

        for p, g, m in zip(params, grads, moments):
            # Learning rate multipliers
            multiplier_name = None
            if self.lr_multipliers:
                multiplier_name = [mult_name for mult_name in self.lr_multipliers
                                   if mult_name in p.name]
            new_lr = self.learning_rate
            if multiplier_name:
                new_lr = new_lr * self.lr_multipliers[multiplier_name[0]]
                if self.init_verbose:
                    print('{} learning rate set for {} -- {}'.format(
                        '%.e' % K.get_value(new_lr), p.name.split('/')[0], new_lr))
            elif not multiplier_name and self.init_verbose:
                print('No change in learning rate {} -- {}'.format(
                    p.name, K.get_value(new_lr)))

            v = self.momentum * m - self.eta_t * new_lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - self.eta_t * new_lr * g
            else:
                new_p = p + v

            # Weight decays
            if p.name in self.weight_decays.keys() and total_iterations != 0:
                wd = self.weight_decays[p.name]
                wd_normalized = wd * K.cast(
                    K.sqrt(self.batch_size / self.total_iterations), 'float32')
                new_p = new_p - self.eta_t * wd_normalized * p
                if self.init_verbose:
                    print('{} weight decay set for {}'.format(
                        K.get_value(wd_normalized), p.name))
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'batch_size': int(K.get_value(self.batch_size)),
                  'total_iterations': int(K.get_value(self.total_iterations)),
                  'weight_decays': self.weight_decays,
                  'lr_multipliers': self.lr_multipliers,
                  'use_cosine_annealing': self.use_cosine_annealing,
                  't_cur': int(K.get_value(self.t_cur)),
                  'eta_t': int(K.get_value(self.eta_t)),
                  'eta_min': int(K.get_value(self.eta_min)),
                  'eta_max': int(K.get_value(self.eta_max)),
                  'init_verbose': self.init_verbose}
        base_config = super(SGDW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
