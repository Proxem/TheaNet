import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

#theano.config.exception_verbosity='high'

inputDim = 1
hiddenDim = 50
nClasses = 1
scale = 0.001

# initial hidden state
h0   = theano.shared(numpy.zeros((hiddenDim, 1), dtype=theano.config.floatX), 'h0')

# reset gate layer
Wr  = theano.shared(scale * numpy.random.uniform(-1.0, 1.0, (hiddenDim, inputDim)).astype(theano.config.floatX), 'Wr')
Ur  = theano.shared(scale * numpy.eye(hiddenDim).astype(theano.config.floatX), 'Ur')
br  = theano.shared(scale * numpy.random.uniform(-1.0, 1.0, (hiddenDim, 1)).astype(theano.config.floatX), 'br')

# update gate layer
Wz  = theano.shared(scale * numpy.random.uniform(-1.0, 1.0, (hiddenDim, inputDim)).astype(theano.config.floatX), 'Wz')
Uz  = theano.shared(scale * numpy.eye(hiddenDim).astype(theano.config.floatX), 'Uz')
bz  = theano.shared(scale * numpy.random.uniform(-1.0, 1.0, (hiddenDim, 1)).astype(theano.config.floatX), 'bz')

# layers
W  = theano.shared(scale * numpy.random.uniform(-1.0, 1.0, (hiddenDim, inputDim)).astype(theano.config.floatX), 'W')
U  = theano.shared(scale * numpy.eye(hiddenDim).astype(theano.config.floatX), 'U')
b  = theano.shared(scale * numpy.random.uniform(-1.0, 1.0, (hiddenDim, 1)).astype(theano.config.floatX), 'b')

# prediction layer
S  = theano.shared(scale * numpy.random.uniform(-1.0, 1.0, (nClasses, hiddenDim)).astype(theano.config.floatX), 'S')
Sb = theano.shared(scale * numpy.random.uniform(-1.0, 1.0, (nClasses, 1)).astype(theano.config.floatX), 'Sb')

#eps  = theano.shared(scale * numpy.ones(1).astype(theano.config.floatX), 'eps') * 0.0001

# bundle
params = [ h0, Wr, Ur, br, Wz, Uz, bz, W, U, b, S, Sb ]

# Adagrad shared variables
hists = {}
for param in params:
    hists[param.name + 'Hist'] = theano.shared(numpy.zeros_like(param.get_value()))

x = T.tensor3('x')
expected = T.matrix('expected')

def recurrence(x_t, h_tm1):
    # reset gate
    r_t = T.nnet.sigmoid(T.dot(Wr, x_t) + T.dot(Ur, h_tm1) + br)
    # update gate
    z_t = T.nnet.sigmoid(T.dot(Wz, x_t) + T.dot(Uz, h_tm1) + bz)
    # proposed hidden state
    _h_t = T.tanh(T.dot(W, x_t) + T.dot(U, r_t * h_tm1) + b)
    # actual hidden state
    h_t = z_t * h_tm1 + (1 - z_t) * _h_t
    return h_t

h, _ = theano.scan(fn=recurrence, sequences=x, outputs_info=h0)

lr = T.scalar('lr')

output = T.dot(S, h[-1]) + Sb
error = 0.5 * T.sum((output - expected) ** 2)
print "gradients"
gradients = T.grad(error, params)

lr = T.scalar('lr')

print "updates"
updates = OrderedDict()
for (param, grad) in zip(params, gradients):
    hist = hists[param.name + "Hist"]
    updates[hist] = hist + grad * grad
    updates[param] = param - lr * grad / T.sqrt(hist + 1e-5)

# theano functions
print "classify"
classify = theano.function(inputs=[x], outputs=output)

print "train"
train = theano.function(inputs=[x, expected, lr],
							  outputs = error,
							  updates = updates )

print "loop"
globalError = 0
for count in xrange(1000000):
    input = numpy.random.uniform(0.0, 1.0, (4, 1, 1)) >= 0.5
    xor = False
    for i in xrange(input.shape[0]):
        xor = xor ^ input[i, 0, 0]
    output = numpy.array([[1.]]) if xor else numpy.array([[-1.]])
    input = input.astype(theano.config.floatX) * 2 - 1

    localError = train(input, output, 0.01)
    globalError += localError
    if count % 1000 == 0:
        print globalError / 1000
        globalError = 0
