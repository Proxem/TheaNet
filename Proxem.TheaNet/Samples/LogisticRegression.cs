/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Collections.Specialized;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Proxem.NumNet;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Samples
{
    /// <summary>
    /// Multi-class Logistic Regression Class
    /// http://deeplearning.net/tutorial/logreg.html
    /// The logistic regression is fully described by a weight matrix :math:`W`
    /// and bias vector :math:`b`. Classification is done by projecting data
    /// points onto a set of hyperplanes, the distance to which is used to
    /// determine a class membership probability.
    /// </summary>
    public class LogisticRegression
    {
        public readonly Tensor<float>.Shared W;
        public readonly Tensor<float>.Shared b;
        public readonly Tensor<float> p_y_given_x;
        public readonly Tensor<int> y_pred;
        public readonly Tensor<float>.Shared[] @params;

        /// <summary>
        /// Initialize the parameters of the logistic regression
        /// </summary>
        /// <param name="input">symbolic variable that describes the input of the architecture(one minibatch)</param>
        /// <param name="n_in">number of input units, the dimension of the space in which the datapoints lie</param>
        /// <param name="n_out">number of output units, the dimension of the space in which the labels lie</param>
        public LogisticRegression(Tensor<float>.Var input, int n_in, int n_out)
        {
            // initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            this.W = T.Shared(NN.Zeros<float>(n_in, n_out), "W");
            // initialize the biases b as a vector of n_out 0s
            this.b = T.Shared(NN.Zeros<float>(n_out), "b");

            // symbolic expression for computing the matrix of class-membership probabilities
            // Where:
            // W is a matrix where column-k represent the separation hyper plain for class-k
            // x is a matrix where row-j  represents input training sample-j
            // b is a vector where element-k represent the free parameter of hyper plain-k
            //this.p_y_given_x = T.Softmax(T.Dot(input, this.W) + this.b.DimShuffle('x', 0));
            this.p_y_given_x = T.Softmax(T.Dot(input, this.W) + b.DimShuffle('x', 0));

            // symbolic description of how to compute prediction as class whose probability is maximal
            this.y_pred = T.Argmax(this.p_y_given_x, axis: 1);

            // parameters of the model
            this.@params = new[] { this.W, this.b };
        }

        /// <summary>
        /// Return the mean of the negative log-likelihood of the prediction of this model under a given target distribution.
        /// </summary>
        /// <param name="y">corresponds to a vector that gives for each example the correct label</param>
        /// <returns></returns>
        public Scalar<float> NegativeLogLikelihood(Tensor<int> y)
        {
            // y.shape[0] is (symbolically) the number of rows in y, i.e., number of examples (call it n) in the minibatch
            // T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
            // T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP) with one row per example and one column per class
            // LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]]
            // and T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch examples) of the elements in v,
            // i.e., the mean log-likelihood across the minibatch.
            return -T.Mean(T.Log(this.p_y_given_x)[T.Range(y.Shape[0]), y]);
        }

        public Scalar<float> Errors(Tensor<int> y)
        {
            // check if y has same dimension of y_pred
            if (y.NDim != this.y_pred.NDim)
                throw new RankException("y should have the same shape as self.y_pred");

            // the T.neq operator returns a vector of 0s and 1s, where 1 represents a mistake in prediction
            return T.Mean(T.Neq(this.y_pred, y));
        }

        public static Tuple<Tensor<float>.Shared, Tensor<int>.Shared>[] LoadData(string path)
        {
            Trace.WriteLine("... loading data");

            var train_set_x = NN.LoadText<float>(Path.Combine(path, "train_set_x.txt"));
            var train_set_y = NN.LoadText<int>(Path.Combine(path, "train_set_y.txt"))[Slicer._, 0];

            var test_set_x = NN.LoadText<float>(Path.Combine(path, "test_set_x.txt"));
            var test_set_y = NN.LoadText<int>(Path.Combine(path, "test_set_y.txt"))[Slicer._, 0];

            var valid_set_x = NN.LoadText<float>(Path.Combine(path, "valid_set_x.txt"));
            var valid_set_y = NN.LoadText<int>(Path.Combine(path, "valid_set_y.txt"))[Slicer._, 0];

            return new[] {
                Tuple.Create(T.Shared(train_set_x, "train_set_x"), T.Shared(train_set_y, "train_set_y")),
                Tuple.Create(T.Shared(valid_set_x, "valid_set_x"), T.Shared(valid_set_y, "valid_set_y")),
                Tuple.Create(T.Shared(test_set_x, "test_set_x"), T.Shared(test_set_y, "test_set_y"))
            };
        }

        public static void SgdOptimizationMnist(float learning_rate = 0.13f, int n_epochs = 1000,
            string dataset = "mnist.pkl.gz",
            int batch_size = 600)
        {
            var sets = LoadData(@"\\HYPERION\ProxemData\R&D\MnistDigit");
            var train_set_x = sets[0].Item1;
            var train_set_y = sets[0].Item2;
            var valid_set_x = sets[1].Item1;
            var valid_set_y = sets[1].Item2;
            var test_set_x = sets[2].Item1;
            var test_set_y = sets[2].Item2;

            // compute number of minibatches for training, validation and testing
            var n_train_batches = train_set_x.Value.Shape[0] / batch_size;
            var n_valid_batches = valid_set_x.Value.Shape[0] / batch_size;
            var n_test_batches = test_set_x.Value.Shape[0] / batch_size;

            ////////////////////////////////////////////
            // BUILD ACTUAL MODEL //
            ////////////////////////////////////////////
            Trace.WriteLine("... building the model");

            // allocate symbolic variables for the data
            var index = T.Scalar<int>("index");  // index to a [mini]batch

            // generate symbolic variables for input (x and y represent a minibatch)
            var x = T.Matrix<float>("x");  // data, presented as rasterized images
            var y = T.Vector<int>("y");  // labels, presented as 1D vector of [int] labels

            // construct the logistic regression class
            // Each MNIST image has size 28*28
            var classifier = new LogisticRegression(input: x, n_in: 28 * 28, n_out: 10);

            // the cost we minimize during training is the negative log likelihood of
            // the model in symbolic format
            var cost = classifier.NegativeLogLikelihood(y);

            // compiling a Theano function that computes the mistakes that are made by
            // the model on a minibatch
            var test_model = T.Function(
                input: index,
                output: classifier.Errors(y),
                givens: new OrderedDictionary {
                    { x, test_set_x[T.Slice(index * batch_size, (index + 1) * batch_size)] },
                    { y, test_set_y[T.Slice(index * batch_size, (index + 1) * batch_size)] }
                }
             );

            var validate_model = T.Function(
                input: index,
                output: classifier.Errors(y),
                givens: new OrderedDictionary {
                    { x, valid_set_x[T.Slice(index * batch_size, (index + 1) * batch_size)] },
                    { y, valid_set_y[T.Slice(index * batch_size, (index + 1) * batch_size)] }
                }
            );

            // compute the gradient of cost with respect to theta = (W,b)
            var g_W = T.Grad(cost: cost, wrt: classifier.W);
            var g_b = T.Grad(cost: cost, wrt: classifier.b);

            // specify how to update the parameters of the model as a list of (variable, update expression) pairs.
            var updates = new OrderedDictionary {
                { classifier.W, classifier.W - learning_rate * g_W },
                { classifier.b, classifier.b - learning_rate * g_b }
            };

            // compiling a Theano function `train_model` that returns the cost, but in
            // the same time updates the parameter of the model based on the rules
            // defined in `updates`
            var train_model = T.Function(
                input: index,
                output: cost,
                updates: updates,
                givens: new OrderedDictionary {
                    { x, train_set_x[T.Slice(index * batch_size, (index + 1) * batch_size)] },
                    { y, train_set_y[T.Slice(index * batch_size, (index + 1) * batch_size)] }
                }
            );

            //////////////////////////////
            // TRAIN MODEL //
            //////////////////////////////
            Trace.WriteLine("... training the model");
            // early-stopping parameters
            var patience = 5000;  // look as this many examples regardless
            var patience_increase = 2;  // wait this much longer when a new best is
                                        // found
            var improvement_threshold = 0.995;  // a relative improvement of this much is
                                                // considered significant
            var validation_frequency = Math.Min(n_train_batches, patience / 2);
            // go through this many
            // minibatche before checking the network
            // on the validation set; in this case we
            // check every epoch

            var best_validation_loss = float.PositiveInfinity;
            var test_score = 0f;
            var timer = Stopwatch.StartNew();

            var done_looping = false;
            var epoch = 0;
            while ((epoch < n_epochs) && (!done_looping))
            {
                epoch = epoch + 1;
                for (var minibatch_index = 0; minibatch_index < n_train_batches; minibatch_index++)
                {
                    var minibatch_avg_cost = train_model(minibatch_index);
                    // iteration number
                    var iter = (epoch - 1) * n_train_batches + minibatch_index;

                    if ((iter + 1) % validation_frequency == 0)
                    {
                        // compute zero-one loss on validation set
                        var validation_losses = Enumerable.Range(0, n_valid_batches).Select(i => validate_model(i));
                        var this_validation_loss = validation_losses.Sum() / n_valid_batches;

                        Trace.WriteLine(string.Format("epoch {0}, minibatch {1}/{2}, validation error {3} %",
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100f
                        ));

                        // if we got the best validation score until now
                        if (this_validation_loss < best_validation_loss)
                        {
                            //improve patience if loss improvement is good enough
                            if (this_validation_loss < best_validation_loss * improvement_threshold)
                            {
                                patience = Math.Max(patience, iter * patience_increase);
                            }

                            best_validation_loss = this_validation_loss;
                            // test it on the test set

                            var test_losses = Enumerable.Range(0, n_test_batches).Select(i => test_model(i));
                            test_score = test_losses.Sum() / n_test_batches;

                            Trace.WriteLine(string.Format("     epoch {0}, minibatch {1}/{2}, test error of best model {3} %",
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_score * 100f
                            ));
                        }
                    }
                    if (patience <= iter)
                    {
                        done_looping = true;
                        break;
                    }
                }
            }
            timer.Stop();
            Trace.WriteLine(string.Format("Optimization complete with best validation score of {0} %, with test performance {1} %",
                best_validation_loss * 100f, test_score * 100f));
            Trace.WriteLine(string.Format("The code run for {0} epochs, with {1} epochs/sec",
                epoch, 1f * epoch / (timer.ElapsedMilliseconds / 1000)));
            Trace.WriteLine(string.Format("The code for file {0} ran for {1} s", "file", timer.ElapsedMilliseconds / 1000));
        }
    }
}
