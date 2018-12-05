import numpy
import theano
import theano.tensor as T


def theano_softmax():
    x = T.tensor3('x')
    _y = T.nnet.softmax(x)
    f = theano.function([x], _y)
    return f


def theano_p_y_given_x():
    x = T.dmatrix('x')
    w = T.dmatrix('w')
    b = T.dmatrix('b')
    input = T.dot(x, w) + b
    y = T.nnet.softmax(input)
    f = theano.function([x, w, b], y)

    return f


def softmax(w):
    w = numpy.array(w)

    maxes = numpy.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = numpy.exp(w - maxes)
    dist = e / numpy.sum(e, axis=1)

    return dist


def p_y_given_x(X, w, b):
    dt = numpy.dot(X, w) + b
    return softmax(dt)


#X = numpy.array([[1, 3], [2, 5]])
X = numpy.array([[[1, 3]]])
w = numpy.array([[1, 1], [1, 1]])
b = numpy.array([[0, 0], [0, 0]])

print "---------------------"
print "Theano"
print theano_softmax()(X)
print "Ours"
print softmax(X)
print "---------------------"
print ""
print "---------------------"
print "Theano P(y) given X:"
print theano_p_y_given_x()(X, w, b)
print "Our P(y) given X:"
print p_y_given_x(X, w, b)
