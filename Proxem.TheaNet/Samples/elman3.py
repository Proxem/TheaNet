import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

nh = 5
nc = 4
#ne = 200000
de = 300
cs = 10
'''
nh :: dimension of the hidden layer
nc :: number of classes
ne :: number of word embeddings in the vocabulary
de :: dimension of the word embeddings
cs :: word window context size 
'''
# parameters of the model
#emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
#		   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
		   (de * cs, nh)).astype(theano.config.floatX))
Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
		   (nh, nh)).astype(theano.config.floatX))
W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
		   (nh, nc)).astype(theano.config.floatX))
bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

# bundle
params = [ Wx, Wh, W, bh, b, h0 ]
names  = ['Wx', 'Wh', 'W', 'bh', 'b', 'h0']
#idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
#x = emb[idxs].reshape((idxs.shape[0], de*cs))
x = T.matrix('x')
#joc: idxs.shape = [sentence, cs], emb.shape = [ne, de], emb[idx].shape = [sentence, cs, de], reshape = [sentence, de * cs] 
y    = T.iscalar('y') # label

def recurrence(x_t, h_tm1):
	h_t = T.nnet.sigmoid(T.dot(x_t, Wx) + T.dot(h_tm1, Wh) + bh)
	s_t = T.nnet.softmax(T.dot(h_t, W) + b)
	return [h_t, s_t]

[h, s], _ = theano.scan(fn=recurrence, \
	sequences=x, outputs_info=[h0, None], \
	n_steps=x.shape[0])

p_y_given_x_lastword = s[-1,0,:]
p_y_given_x_sentence = s[:,0,:]
y_pred = T.argmax(p_y_given_x_sentence, axis=1)

# cost and gradients and learning rate
lr = T.scalar('lr')
nll = -T.mean(T.log(p_y_given_x_lastword)[y])
gradients = T.grad( nll, params )
updates = OrderedDict(( p, p-lr*g ) for p, g in zip( params , gradients))

# theano functions
classify = theano.function(inputs=[x], outputs=y_pred)

train = theano.function( inputs  = [x, y, lr],
							  outputs = nll,
							  updates = updates )

normalize = theano.function( inputs = [],
				 updates = {emb:\
				 emb/T.sqrt((emb**2).sum(axis=1)).dimshuffle(0,'x')})
