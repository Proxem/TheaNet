import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

nh = 3
Wbit  = theano.shared(0.5 * numpy.random.uniform(-1.0, 1.0, (nh, 1)).astype(theano.config.floatX))
Wstate  = theano.shared(0.5 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))
Wout   = theano.shared(0.5 * numpy.random.uniform(-1.0, 1.0, (1, nh)).astype(theano.config.floatX))
b   = theano.shared(0.5 * numpy.random.uniform(-1.0, 1.0, (nh, 1)).astype(theano.config.floatX))

state0   = theano.shared(numpy.zeros((nh, 1), dtype=theano.config.floatX))

# bundle
params = [ Wbit, Wstate, Wout, b, state0 ]

bits = T.tensor3('bits')
expected = T.matrix('expected')

def recurrence(bit, oldState):
	return T.nnet.sigmoid(T.dot(Wbit, bit) + T.dot(Wstate, oldState) + b)

states, _ = theano.scan(fn=recurrence, \
	sequences=bits, outputs_info=state0)

output = T.dot(Wout, states[-1])
error = 0.5 * T.sum((output - expected) ** 2)

gradients = T.grad(error, params)
lr = T.scalar('lr')
updates = OrderedDict(( p, p-lr*g ) for p, g in zip(params , gradients))

# theano functions
print "classify"
classify = theano.function(inputs=[bits], outputs=output)

print "train"
train = theano.function(inputs=[bits, expected, lr],
							  outputs = error,
							  updates = updates )

print "loop"
globalError = 0
for count in xrange(1000000):
    input = numpy.random.uniform(0.0, 1.0, (2, 1, 1)) >= 0.5
    xor = input[0, 0, 0] ^ input[1, 0, 0]
    output = numpy.array([[1.]]) if xor else numpy.array([[0.]])
    input = input.astype(theano.config.floatX)

    localError = train(input, output, 0.01)
    globalError += localError
    if count % 1000 == 0:
        print globalError / 1000
        globalError = 0
