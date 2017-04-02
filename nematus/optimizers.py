'''
Optimizers
'''

import numpy
from collections import OrderedDict

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from util import *
from theano_util import *

# Calling convention:
# f_grad_shared, f_update = name(hyperp, tparams, grads, inputs (list), cost)
# with profile as an optional argument

def adam(lr, tparams, grads, inp, cost, beta1=0.9, beta2=0.999, e=1e-8, optimizer_params={}, profile=False):
    PREFIX='adam_'

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    updates = []
    optimizer_tparams = {}

    t_prev_name = PREFIX + 't_prev'
    if t_prev_name in optimizer_params:
        t_prev_init = optimizer_params[t_prev_name]
    else:
        t_prev_init = 0.
    t_prev = theano.shared(numpy.float32(t_prev_init), t_prev_name)
    optimizer_tparams[t_prev_name] = t_prev
    
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2**t) / (1. - beta1**t)

    for p, g in zip(tparams.values(), gshared):
        # Create/Load variable for first moment
        m_name = PREFIX + p.name + '_mean'
        if m_name in optimizer_params:
            m_init = optimizer_params[m_name]
        else:
            m_init = p.get_value() * 0.
        m = theano.shared(m_init, m_name)
        optimizer_tparams[m_name] = m

        # Create/Load variable for second moment
        v_name = PREFIX + p.name + '_variance'
        if v_name in optimizer_params:
            v_init = optimizer_params[v_name]
        else:
            v_init = p.get_value() * 0.
        v = theano.shared(v_init, v_name)
        optimizer_tparams[v_name] = v

        # Define updates on shared vars
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g**2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update, optimizer_tparams

def adadelta(lr, tparams, grads, inp, cost, optimizer_params={}, profile=False):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    # TODO: third return value should be a dict of name->shared var used by optimizer
    return f_grad_shared, f_update, {}


def rmsprop(lr, tparams, grads, inp, cost, optimizer_params={}, profile=False):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    # TODO: third return value should be a dict of name->shared var used by optimizer
    return f_grad_shared, f_update, {}

# no place is reserved for alpha, using as a constant for the moment
ALPHA = 0.3
def backstitch_sgd(lr, tparams, grads, inp, cost, optimizer_params=None, profile=False):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup,
                                    profile=profile)

    pup1 = [(p, p + ALPHA * lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update1 = theano.function([lr], [], updates=pup1, profile=profile)

    pup2 = [(p, p - (1 + ALPHA) * lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update2 = theano.function([lr], [], updates=pup2, profile=profile)

    return f_grad_shared, f_update1, f_update2, {}

def backstitch_adadelta(lr, tparams, grads, inp, cost, optimizer_params={}, profile=False):
    gshared = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    gsup = [(zg, g) for zg, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inp, cost, updates=gsup,
                                    profile=profile)

    adlr = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6)
             for ru2, rg2 in zip(running_up2, running_grads2)] # adadelta learning rate
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, gshared)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * ((nu * g) ** 2))
             for ru2, g in zip(running_up2, gshared)]
    pup1 = [(p, p + ALPHA * nu * g) for p, nu in zip(itemlist(tparams), adlr)]
    pup2 = [(p, p - (1 + ALPHA) * nu * g) for p, nu in zip(itemlist(tparams), adlr)]

    f_update1 = theano.function([lr], [], updates=rg2up + ru2up + pup1,
                               on_unused_input='ignore', profile=profile)
    f_update2 = theano.function([lr], [], updates=pup2,
                               on_unused_input='ignore', profile=profile)

    # TODO: third return value should be a dict of name->shared var used by optimizer
    return f_grad_shared, f_update1, f_update2, {}

F_ALPHA = ALPHA + ALPHA * ALPHA
F_EPSILON = 0.1
def fast_backstitch_sgd(lr, tparams, grads, inp, cost, optimizer_params=None, profile=False):
    dshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    dsup1 = [(ds, ds + g) for ds, g in zip(dshared, grads)] # accumulate
    dsup2 = [(ds, -ds * (F_ALPHA / F_EPSILON - 1 - F_EPSILON) / (F_ALPHA / F_EPSILON)) for ds in dshared] # scale
    dsup3 = [(ds, ds * 0.) for ds in dshared] # clear

    f_grad_shared = theano.function(inp, cost, updates=dsup1,
                                    profile=profile)

    pup1 = [(p, p + F_EPSILON * lr * d) for p, d in zip(itemlist(tparams), dshared)]
    f_update1 = theano.function([lr], [], updates=pup1 + dsup2, profile=profile)

    pup2 = [(p, p - F_ALPHA / F_EPSILON * lr * g) for p, g in zip(itemlist(tparams), dshared)]
    f_update2 = theano.function([lr], [], updates=pup2 + dsup3, profile=profile)

    return f_grad_shared, f_update1, f_update2, {}

def backstitch_adadelta(lr, tparams, grads, inp, cost, optimizer_params={}, profile=False):
    dshared = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    dsup1 = [(ds, ds + g) for ds, g in zip(gshared, grads)]
    dsup2 = [(ds, -ds * (F_ALPHA / F_EPSILON - 1 - F_EPSILON) / (F_ALPHA / F_EPSILON)) for ds in dshared] # scale
    dsup3 = [(ds, ds * 0.) for ds in dshared] # clear

    f_grad_shared = theano.function(inp, cost, updates=dsup1,
                                    profile=profile)

    adlr = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6)
             for ru2, rg2 in zip(running_up2, running_grads2)] # adadelta learning rate
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * ((nu * g) ** 2))
             for ru2, g in zip(running_up2, grads)]
    pup1 = [(p, p + F_EPSILON * nu * g) for p, nu in zip(itemlist(tparams), adlr)]
    pup2 = [(p, p - F_ALPHA / F_EPSILON * nu * g) for p, nu in zip(itemlist(tparams), adlr)]

    f_update1 = theano.function([lr], [], updates=rg2up + ru2up + pup1 + dsup2,
                               on_unused_input='ignore', profile=profile)
    f_update2 = theano.function([lr], [], updates=pup2 + dsup3,
                               on_unused_input='ignore', profile=profile)

    # TODO: third return value should be a dict of name->shared var used by optimizer
    return f_grad_shared, f_update1, f_update2, {}

def sgd(lr, tparams, grads, inp, cost, optimizer_params=None, profile=False):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update, {}

