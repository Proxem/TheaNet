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
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Proxem.NumNet;
using Proxem.TheaNet.Samples;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TrainingTest
    {
        System.Random random;

        [TestInitialize]
        public void Initialize()
        {
            Runtime.Reset();
            NN.Random.Seed(15677);
            random = new System.Random(12350);
        }

        [TestMethod]
        public void TestFeedforward()
        {
            var xor = new[]
            {
                (new [] { 0f, 0f }, new [] { 0f }),
                (new [] { 0f, 1f }, new [] { 1f }),
                (new [] { 1f, 0f }, new [] { 1f }),
                (new [] { 1f, 1f }, new [] { 0f }),
            };

            var ff = new NeuralNet.Network(2, 2, 1);
            foreach (var error in ff.Backprop(0.1f, 0.001f, 10000, xor))
            {

            }
        }

        [TestMethod, TestCategory("Slow")]
        public void TestSimple()
        {
            var xor = new[]
            {
                (NN.Array(0f, 0f), 0),
                (NN.Array(0f, 1f), 1),
                (NN.Array(1f, 0f), 1),
                (NN.Array(1f, 1f), 0),
            };

            var ff = new Simple(2, 3, 2, 1f);
            float globalError = 0;
            float globalScore = 0;
            for (int count = 0; count < 1000; count++)
            {
                NN.Random.Shuffle(xor);
                int score = 0;
                foreach (var sample in xor)
                {
                    var nll = ff.train(sample.Item1, sample.Item2, 0.1f);
                    var pred = ff.classify(sample.Item1);
                    score = NN.Argmax(pred) == sample.Item2 ? 1 : 0;
                    globalError += nll;
                    globalScore += score;
                }
                if (count % 1000 == 0)
                {
                    int n = count * xor.Length;
                    Trace.WriteLine($"error: {globalError / n }, accuracy: {globalScore * 100 / n }%");
                    globalError = 0;
                    globalScore = 0;
                }
            }
        }

        [TestMethod]
        public void TestSimpleCompiles()
        {
            var ff = new Simple(2, 3, 2, 1f);
            ff.train(NN.Array(0f, 1f), 1, 0.1f);
            ff.classify(NN.Array(0f, 1f));
        }

        private bool Contains111_2d(Array<float> a)
        {
            int count = 0;
            for (int i = 0; i < a.Shape[0]; ++i)
                for (int j = 0; j < a.Shape[1]; ++j)
                {
                count = (a.Item[i,j] == 1) ? count + 1 : 0;
                if (count >= 3) return true;
                }
            for (int j = 0; j < a.Shape[1]; ++j)
                for (int i = 0; i < a.Shape[0]; ++i)
                {
                    count = (a.Item[i, j] == 1) ? count + 1 : 0;
                    if (count >= 3) return true;
                }
            return false;
        }

        private bool Contains111(Array<float> a)
        {
            int count = 0;
            for (int i = 0; i < a.Shape[0]; ++i)
            {
                count = (a.Item[i] == 1) ? count + 1 : 0;
                if (count >= 3) return true;
            }
            return false;
        }

        [TestMethod]
        public void SimpleCNNTrains()
        {
            int inputLength = 10;
            int kernelSize = 3;
            float c = 0;
            var ff = new SimpleCNN(inputLength, kernelSize, 2, 1f);
            float globalError = 0;
            float globalScore = 0;
            for (int count = 0; count < 100000; count++)
            {
                var sample = NN.Random.Bernoulli(0.3, inputLength).As<float>();
                var y = Contains111(sample) ? 1 : 0;
                var nll = ff.train(sample, y, 0.05f);
                var pred = ff.classify(sample);
                var score = NN.Argmax(pred) == y ? 1 : 0;
                globalError += nll / 1000f;
                globalScore += score / 1000f;
                c += y / 1000f;
                if (count % 1000 == 0)
                {
                    Trace.WriteLine(string.Format("{0} {1} {2} {3} %", globalError, globalScore * 100, c * 100, pred));
                    globalError = 0;
                    globalScore = 0;
                    c = 0;
                }
            }

            AssertArray.IsGreaterThan(globalScore, 0.95);
            AssertArray.IsLessThan(globalError, 0.25);
        }

        [TestMethod, TestCategory("Slow")]
        public void TestSimpleCNN2d()
        {
            var inputShape = new [] { 100, 100 };
            var kernelShape = new [] { 5, 5 };
            var poolingShape = new [] { 96, 96 };

            var ff = new SimpleCNN2d(inputShape, kernelShape, poolingShape, 2);
            float c = 0;
            float globalError = 0;
            float globalScore = 0;
            for (int count = 0; count < 100000; count++)
            {
                var sample = NN.Random.Bernoulli(0.027, inputShape).As<float>();
                var y = Contains111_2d(sample) ? 1 : 0;
                var nll = ff.train(sample, y, 0.05f);
                int pred = ff.classify(sample).Values[0];
                var score = pred == y ? 1 : 0;
                //var debug = ff.debug(sample);
                globalError += nll / 1000f;
                globalScore += score / 1000f;
                //debug = NN.Softmax(debug.T);
                //Trace.WriteLine(string.Format("true : {0}, pred :  {1}, score : {2}", y, pred, score));
                c += y / 1000f;
                if (count % 1000 == 0)
                {
                    //Trace.WriteLine(string.Format("{0}", debug));
                    Trace.WriteLine(string.Format("{0} {1} % {2} %", globalError, globalScore * 100, c * 100));
                    globalError = 0;
                    globalScore = 0;
                    c = 0;
                }
            }

            AssertArray.IsGreaterThan(globalScore, 0.95);
            AssertArray.IsLessThan(globalError, 0.25);
        }

        [TestMethod, TestCategory("Slow")]
        public void TestSimpleYoonKim()
        {
            var inputShape = new int[2] { 10, 10 };
            var kernelShape = new int[2] { 3, 3 };
            var poolingShape = new int[2] { 2, 2 };

            var ff = new SimpleCNN2d(inputShape, kernelShape, poolingShape, 2);
            float c = 0;
            float globalError = 0;
            float globalScore = 0;
            for (int count = 0; count < 100000; count++)
            {
                var sample = NN.Random.Bernoulli(0.15, inputShape).As<float>();
                var y = Contains111_2d(sample) ? 1 : 0;
                var nll = ff.train(sample, y, 0.05f);
                int pred = ff.classify(sample).Values[0];
                var score = pred == y ? 1 : 0;
                //var debug = ff.debug(sample);
                globalError += nll / 1000f;
                globalScore += score / 1000f;
                //debug = NN.Softmax(debug.T);
                //Trace.WriteLine(string.Format("true : {0}, pred :  {1}, score : {2}", y, pred, score));
                c += y / 1000f;
                if (count % 1000 == 0)
                {
                    //Trace.WriteLine(string.Format("{0}", debug));
                    Trace.WriteLine(string.Format("{0} {1} % {2} %", globalError, globalScore * 100, c * 100));
                    globalError = 0;
                    globalScore = 0;
                    c = 0;
                }
            }

            AssertArray.IsGreaterThan(globalScore, 0.95);
            AssertArray.IsLessThan(globalError, 0.25);
        }


        [TestMethod, TestCategory("Slow")]
        public void TestElman()
        {
            // http://matpalm.com/blog/2015/03/28/theano_word_embeddings/

            const int nh = 30;    // dimension of the hidden layer
            const int nc = 2;     // number of classes
            const int ne = 3000;  // number of word embeddings in the vocabulary
            const int de = 50;    // dimension of the word embeddings
            const int cs = 3;     // word window context size

            var elman = new Elman(nh, nc, ne, de, cs);

            var sequence = NN.Array(new int[4, cs] { { 1, 2, 3 }, { 3, 4, 5 }, { 10, 11, 12 }, { 2, 4, 6 } });

            elman.classify(sequence);
            var y = 1;      // 0 <= y < nc
            elman.train(sequence, y, 0.01f);
        }

        [TestMethod, TestCategory("Slow")]
        public void TestJordan()
        {
            // http://matpalm.com/blog/2015/03/28/theano_word_embeddings/

            const int nh = 30;    // dimension of the hidden layer
            const int nc = 2;     // number of classes
            const int ne = 3000;  // number of word embeddings in the vocabulary
            const int de = 50;    // dimension of the word embeddings
            const int cs = 3;     // word window context size

            var jordan = new Jordan(nh, nc, ne, de, cs);

            var sequence = NN.Array(new int[4, cs] { { 1, 2, 3 }, { 3, 4, 5 }, { 10, 11, 12 }, { 2, 4, 6 } });

            jordan.classify(sequence);
            var y = 1;      // 0 <= y < nc
            jordan.train(sequence, y, 0.01f);
        }

        [TestMethod, TestCategory("Slow")]
        public void TestElman2()
        {
            // http://matpalm.com/blog/2015/03/28/theano_word_embeddings/

            //var elman = new Elman(nh: 300, nc: 4, ne: 200000, de: 300, cs: 10);
            var elman = new Elman2(nh: 30, nc: 2, de: 1, cs: 1);

            var xor = new[]
            {
                (new [] { 0f, 0f, 0f }, 0),
                (new [] { 0f, 0f, 1f }, 1),
                (new [] { 0f, 1f, 0f }, 1),
                (new [] { 0f, 1f, 1f }, 0),
            };

            var lr = 0.01f;
            float globalError = 0;
            for (int count = 0; count < 100; ++count)
            {
                var x = NN.Array<float>(xor[count % xor.Length].Item1).Reshape(-1, 1, 1);
                int y = xor[count % xor.Length].Item2;
                var error = elman.train(x, y, lr);
                globalError += error;
                if (count % 1000 == 0)
                {
                    Trace.WriteLine(globalError / 1000f);
                    globalError = 0;
                }
            }
        }

        [TestMethod]
        public void Elman2Compiles()
        {
            var elman = new Elman2(nh: 30, nc: 2, de: 1, cs: 1);
            var x = NN.Array(new[] { 0f, 1f, 0f }).Reshape(-1, 1, 1);
            elman.train(x, 1, 0.01f);
        }

        [TestMethod/*, TestCategory("Slow")*/]
        public void TestElman3()
        {
            // http://matpalm.com/blog/2015/03/28/theano_word_embeddings/
            var elman = new Elman3(nh: 30, nc: 2, de: 1, cs: 1);

            var xor = new[]
            {
                (new [] { 0f, 0f, 0f }, 0),
                (new [] { 0f, 0f, 1f }, 1),
                (new [] { 0f, 1f, 0f }, 1),
                (new [] { 0f, 1f, 1f }, 0),
            };

            var expected = new[] { 0.8451504f, 0.785709f, 0.7685911f, 0.7557927f, 0.7443435f, 0.7313102f, 0.7108455f, 0.6661955f, 0.5576075f, 0.3245486f, 0.08225087f, 0.02822098f, 0.0153197f, 0.01012096f, 0.007412266f };
            var actual = new float[expected.Length];
            var lr = 0.1f;
            float globalError = 0;
            for (int count = 1; count <= expected.Length * 1000; ++count)
            {
                var x = NN.Array<float>(xor[count % xor.Length].Item1).Reshape(-1, 1);
                int y = xor[count % xor.Length].Item2;
                var error = elman.train(x, y, lr);
                globalError += error;
                if (count % 1000 == 0)
                {
                    int i = count / 1000 - 1;
                    var avgError = globalError / 1000;
                    actual[i] = avgError;
                    Trace.WriteLine(avgError);
                    globalError = 0;
                }
            }
            AssertArray.AreAlmostEqual(expected, actual, relativeErr: 1e-5f);
        }

        [TestMethod, TestCategory("Slow")]
        public void TestJordan2()
        {
            // http://matpalm.com/blog/2015/03/28/theano_word_embeddings/
            var jordan = new Jordan2(nh: 30, nc: 2, de: 1, cs: 1);

            var xor = new[]
            {
                (new [] { 0f, 0f, 0f }, 0),
                (new [] { 0f, 0f, 1f }, 1),
                (new [] { 0f, 1f, 0f }, 1),
                (new [] { 0f, 1f, 1f }, 0),
            };

            var lr = 0.1f;
            float globalError = 0;
            for (int count = 1; count <= 100000; ++count)
            {
                var x = NN.Array<float>(xor[count % xor.Length].Item1).Reshape(-1, 1);
                int expected = xor[count % xor.Length].Item2;
                var error = jordan.train(x, expected, lr);
                globalError += error;
                if (count % 1000 == 0)
                {
                    Trace.WriteLine((float)(globalError / 1000));
                    globalError = 0;
                }
            }
        }

        [TestMethod]
        public void Jordan2Compiles()
        {
            var jordan = new Jordan2(nh: 30, nc: 2, de: 1, cs: 1);
            var x = NN.Array(new[] { 0f, 1f, 0f }).Reshape(-1, 1);
            jordan.train(x, 1, 0.1f);
        }


        [TestMethod, TestCategory("Slow")]
        public void TestRnn1()
        {
            var xor = new[]
            {
                (new [] { 0f, 0f }, new [] { 0f }),
                (new [] { 0f, 1f }, new [] { 1f }),
                (new [] { 1f, 0f }, new [] { 1f }),
                (new [] { 1f, 1f }, new [] { 0f }),
            };

            var nh = 3; // hidden layer

            var Wbit = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, 1).As<float>(), "Wbit");
            var Wstate = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, nh).As<float>(), "Wstate");
            var Wout = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 1, nh).As<float>(), "Wout");
            var state0 = T.Shared(NN.Zeros<float>(nh, 1), "state0");

            Func<Tensor<float>, Tensor<float>, Tensor<float>> recurrence = (bit, oldState) =>
                T.Tanh(T.Dot(Wbit, bit) + T.Dot(Wstate, oldState));

            var bit1 = T.Matrix<float>("bit1");
            var bit2 = T.Matrix<float>("bit2");
            var expected = T.Matrix<float>("expected");        // 1 x 1

            var test = recurrence(bit2, recurrence(bit1, state0));
            test = T.Dot(Wout, test);
            //test = T.Sigmoid(T.Dot(this.Wout, test));
            var e = 0.5f * T.Norm2(test - expected);
            var d = T.Grad(e);
            var updates = new OrderedDictionary();
            foreach (var W in d.Keys)
                updates[W] = W - 0.001f * d[W];
            var train = T.Function(input: (bit1, bit2, expected), output: e, updates: updates);

            var input1 = NN.Array<float>(xor[0].Item1[0]).Reshape(1, 1);
            var input2 = NN.Array<float>(xor[0].Item1[1]).Reshape(1, 1);
            var output = NN.Array<float>(xor[0].Item2).Reshape(1, 1);
            var error = train(input1, input2, output);
        }

        [TestMethod, TestCategory("Slow")]
        public void TestRnn2()
        {
            var xor = new[]
            {
                (new [] { 0f, 0f }, new [] { 0f }),
                (new [] { 0f, 1f }, new [] { 1f }),
                (new [] { 1f, 0f }, new [] { 1f }),
                (new [] { 1f, 1f }, new [] { 0f }),
            };

            var nh = 3; // hidden layer

            var Wbit = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 1, nh).As<float>(), "Wbit");
            var Wstate = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, nh).As<float>(), "Wstate");
            var Wout = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, 1).As<float>(), "Wout");
            var state0 = T.Shared(NN.Zeros<float>(1, nh), "state0");

            // bundle
            var @params = new[] { Wbit, Wstate, Wout };     // temp: bug in computing h0

            Func<Tensor<float>, Tensor<float>, Tensor<float>> recurrence = (bit, oldState) =>
            {
                return T.Tanh(T.Dot(bit, Wbit) + T.Dot(oldState, Wstate));
            };

            var bit1 = T.Matrix<float>("bit1");
            var bit2 = T.Matrix<float>("bit2");
            var expected = T.Matrix<float>("expected");        // 1 x 1

            var test = recurrence(bit2, recurrence(bit1, state0));
            test = T.Dot(test, Wout);
            //test = T.Sigmoid(T.Dot(this.Wout, test));
            var e = 0.5f * T.Norm2(test - expected);
            var d = T.Grad(e);
            var updates = new OrderedDictionary();
            foreach (var W in d.Keys)
                updates[W] = W - 0.001f * d[W];
            var train = T.Function(input: (bit1, bit2, expected), output: e, updates: updates);

            var input1 = NN.Array<float>(xor[0].Item1[0]).Reshape(1, 1);
            var input2 = NN.Array<float>(xor[0].Item1[1]).Reshape(1, 1);
            var output = NN.Array<float>(xor[0].Item2).Reshape(1, 1);
            var error = train(input1, input2, output);
        }


        [TestMethod, TestCategory("Slow")]
        public void RnnXor()
        {
            float targetErr = 0.01f;
            int nh = 10; // hidden layer

            var Wbit = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, 1).As<float>(), "Wbit");
            var Wstate = T.Shared(NN.Eye<float>(nh), "Wstate");
            var Wout = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 1, nh).As<float>(), "Wout");
            var b = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, 1).As<float>(), "b");

            var state0 = T.Shared(NN.Zeros<float>(nh, 1), "state0");

            // bundle
            var @params = new[] { Wbit, Wstate, Wout, b, state0 };

            var bits = T.Tensor3<float>("bits");               // n x 1
            var expected = T.Matrix<float>("expected");        // 1 x 1

            Func<Tensor<float>, Tensor<float>, Tensor<float>> recurrence = (bit, oldState) =>
            {
                return T.Tanh(T.Dot(Wbit, bit) + T.Dot(Wstate, oldState) + b);
            };

            var states = T.Scan(fn: recurrence,
                sequence: bits, outputsInfo: state0);

            //var output = T.Tanh(T.Dot(Wout, states[-1]));
            //var output = T.Sigmoid(T.Dot(Wout, states[-1]));
            var output = T.Tanh(T.Dot(Wout, states[-1]));
            var error = 0.5f * T.Norm2(output - expected);

            var classify = T.Function(bits, output);

            var gradients = T.Grad(error);
            var updates = new OrderedDictionary();
            var lr = T.Scalar<float>("lr");

            for (int i = 0; i < @params.Length; i++)
                //updates[momemtum[i]] = 0.99 * momemtum[i] + lr * gradients[@params[i]];
                //updates[@params[i]] = @params[i] - momemtum[i];
                updates[@params[i]] = @params[i] - lr * gradients[@params[i]];

            // theano functions
            var train = T.Function(
                input: (bits, expected, lr),
                output: error,
                updates: updates
            );

            double globalError = 0.0, score = 0.0;
            var input = new Array<float>(3, 1, 1);
            //var input = new Array<float>(4, 1, 1);
            var o = new Array<float>(1, 1);
            for (int count = 1; count < 100000; count++)
            {
                //var input = new Array<float>(2 + random.Next(count / 5000), 1, 1);
                bool result = false;
                for (int i = 0; i < input.Shape[0]; i++)
                {
                    var bit = random.NextDouble() >= 0.5;
                    result = result ^ bit;
                    input.Item[i, 0, 0] = bit ? 1.0f : -1.0f;
                }
                o.Item[0, 0] = result ? 1.0f : -1.0f;

                var localError = train(input, o, 0.01f);
                score += (Math.Sign(classify(input).Item[0, 0]) == Math.Sign(o.Item[0, 0]) ? 1 : 0) / 1000f;
                globalError += localError / 1000f;
                if (count % 1000 == 0)
                {
                    Trace.WriteLine(globalError);
                    var n1 = NN.Norm(Wbit.Value);
                    var n2 = NN.Norm(Wstate.Value);
                    if (globalError < targetErr) break;
                    globalError = 0;
                    score = 0;
                }
            }
            AssertArray.IsGreaterThan(score, 0.95);
            AssertArray.IsLessThan(globalError, targetErr);
        }

        [TestMethod, TestCategory("Slow")]
        public void LstmXor()
        {
            float targetErr = 0.01f;
            int nh = 50; // hidden layer

            var lstm = new GRU(1, nh, 1, scale: 0.001f);

            double globalError = 0.0, score = 0.0;
            var input = new Array<float>(4, 1);
            var y = new Array<float>(1);

            for (int count = 1; count < 100000; count++)
            {
                //var input = new Array<float>(2 + random.Next(count / 5000), 1);
                //var input = new Array<float>(2 + random.Next(8), 1);
                bool result = false;
                for (int i = 0; i < input.Shape[0]; i++)
                {
                    var bit = random.NextDouble() >= 0.5;
                    result = result ^ bit;
                    input.Item[i, 0] = bit ? 1.0f : -1.0f;
                }
                y.Item[0] = result ? 1 : -1;

                var localError = lstm.Train(input, y);
                lstm.Update(0.01f, 1);
                var pred = lstm.Classify(input);
                score += (Math.Sign(pred.Item[0]) == y.Item[0] ? 1 : 0) / 1000f;
                globalError += localError / 1000f;
                if (count % 1000 == 0)
                {
                    Trace.WriteLine(string.Format("{0} {1:F1} %", globalError, score * 100));
                    if (globalError < targetErr) break;
                    globalError = 0;
                    score = 0;
                }
            }
            AssertArray.IsLessThan(0.95, score);
            AssertArray.IsLessThan(globalError, targetErr);
        }

        [TestMethod]
        public void LstmXorCompiles()
        {
            var lstm = new GRU(1, 50, 1, scale: 0.001f);
            var input = NN.Array(new float[] { 0, 1, 1, 0 }).Reshape(-1, 1);
            var y = NN.Array(new float[] { 0 });

            lstm.Train(input, y);
            lstm.Update(0.01f, 1);
            lstm.Classify(input);
        }

        [TestMethod, TestCategory("Slow")]
        public void LstmXor2()
        {
            int nh = 50; // hidden layer

            var expected = new[] { 0.7475675f, 0.1499892f, 0.08312643f, 0.0533f };

            var lstm = new GRU2(1, nh, 2, 1f);

            double globalError = 0;
            var input = new Array<float>(4, 1);
            for (int count = 1; count < expected.Length * 1000; count++)
            {
                bool result = false;
                for (int i = 0; i < input.Shape[0]; i++)
                {
                    var bit = random.NextDouble() >= 0.5;
                    result = result ^ bit;
                    input.Item[i, 0] = bit ? 1.0f : -1.0f;
                }
                int y = result ? 1 : 0;

                var localError = lstm.train(input, y, 0.01f);
                globalError += localError;
                if (count % 1000 == 0)
                {
                    Trace.WriteLine((float)(globalError / 1000));
                    AssertArray.AreAlmostEqual(expected[count / 1000 - 1], (float)(globalError / 1000), 1e-6f);
                    globalError = 0;
                }
            }
        }

        [TestMethod]
        public void LstmXor2Compiles()
        {
            var lstm = new GRU2(1, 50, 1, scale: 0.001f);
            var input = NN.Array(new float[] { 0, 1, 1, 0 }).Reshape(4, 1);
            lstm.train(input, 0, 0.01f);
        }

        [TestMethod, TestCategory("Slow")]
        public void TestLogisticRegression()
        {
            LogisticRegression.SgdOptimizationMnist();
        }

        [TestMethod]
        public void LogisticRegressionCompiles()
        {
            int batch_size = 600;
            float learning_rate = 0.13f;

            var index = T.Scalar<int>("index");
            var x = T.Matrix<float>("x");
            var y = T.Vector<int>("y");

            Trace.WriteLine("... creating model graph");
            var classifier = new LogisticRegression(input: x, n_in: 28 * 28, n_out: 10);
            var cost = classifier.NegativeLogLikelihood(y);

            Trace.WriteLine("... loading data");
            var path = @"\\HYPERION\ProxemData\R&D\MnistDigit";
            var valid_set_x = T.Shared(NN.LoadText<float>(Path.Combine(path, "valid_set_x.txt")), "X");
            var valid_set_y = T.Shared(NN.LoadText<int>(Path.Combine(path, "valid_set_y.txt"))[Slicer._, 0], "Y");

            Trace.WriteLine("... compiling evaluation function");
            var validate_model = T.Function(
                input: index,
                output: classifier.Errors(y),
                givens: new OrderedDictionary {
                    { x, valid_set_x[T.Slice(index * batch_size, (index + 1) * batch_size)] },
                    { y, valid_set_y[T.Slice(index * batch_size, (index + 1) * batch_size)] }
                }
            );

            var g_W = T.Grad(cost: cost, wrt: classifier.W);
            var g_b = T.Grad(cost: cost, wrt: classifier.b);

            var updates = new OrderedDictionary {
                { classifier.W, classifier.W - learning_rate * g_W },
                { classifier.b, classifier.b - learning_rate * g_b }
            };

            Trace.WriteLine("... compiling training function");
            var train_model = T.Function(
                input: index,
                output: cost,
                updates: updates,
                givens: new OrderedDictionary {
                    { x, valid_set_x[T.Slice(index * batch_size, (index + 1) * batch_size)] },
                    { y, valid_set_y[T.Slice(index * batch_size, (index + 1) * batch_size)] }
                }
            );

            Trace.WriteLine("... training the model once");
            train_model(1);
        }

        [TestMethod]
        public void CharCNNCompiles()
        {
            var cnn = new CharCNN(10, 3, 5);
            cnn.classify("bouger");
            cnn.DoOneEpoch(0.1f);
        }

        [TestMethod]
        public void CharCNNPassesGradientCheck()
        {
            var cnn = new CharCNN(10, 3, 5);
            var loss = (Scalar<float>)cnn.Loss.Patch(new Patch {[cnn.gold] = 0 });
            foreach (var W in cnn.@params)
                AssertTensor.PassesGradientCheck(cnn.chars, loss, W);
        }

        [TestMethod]
        public void CharCNNTrains()
        {
            var cnn = new CharCNN(10, 3, 5);
            float lr0 = 0.1f;

            float accTest0 = cnn.TestAccuracy();
            float err = 1000f, accTrain = 0f, accTest = 0f;
            for (int i = 0; i < 100; ++i)
            {
                err = cnn.DoOneEpoch(lr0 / (float)Math.Sqrt(1 + i));
                accTrain = cnn.TrainAccuracy();
                accTest = cnn.TestAccuracy();
                Trace.WriteLine($"One epoch {i}, error: {err}, train accuracy: {accTrain}, test accuracy: {accTest}");
            }

            AssertArray.IsLessThan(err, 0.1f);
            AssertArray.IsGreaterThan(accTest, accTest0);
            Assert.AreEqual(accTrain, 1f);
        }


        [TestMethod]
        [TestCategory("TODO")]
        public void TsneCompiles()
        {
            var X_ = NN.Random.Uniform(-1.2f, -0.8f, 100, 10);
            X_[Slicer.From(70)] = NN.Random.Uniform(0.8f, 1.2f, 30, 10);
            var tsne = new Tsne(X_, 2, 40);
            var loss0 = tsne.Loss();
            tsne.Train();
        }

        [TestMethod, TestCategory("TODO")]
        public void TsneTrains()
        {
            var X_ = NN.Random.Uniform(-1.2f, -0.8f, 100, 10);
            X_[Slicer.From(70)] = NN.Random.Uniform(0.8f, 1.2f, 30, 10);
            var tsne = new Tsne(X_, 2, 40);
            var loss0 = tsne.Loss();
            tsne.Train();
            var loss1 = tsne.Loss();
            AssertArray.IsLessThan(loss1, loss0);
        }

        [TestMethod]
        public void ReinforceMulCompiles()
        {
            var model = new ReinforceMul();
            model.Train(0.1f);
            model.Mul(5, 2);
        }

        [TestMethod, TestCategory("Slow")]
        public void ReinforceMulTrains()
        {
            var model = new ReinforceMul();
            var pred = model.Mul(2, 5);
            var acc = model.TrainFor(500, 100);
            AssertArray.IsGreaterThan(acc, 0.8);
        }

        [TestMethod]
        public void ReinforceConstTrains()
        {
            var model = new ReinforceConst();
            var pred = model.Const(2);
            model.TrainFor(100, 100);
            var acc = model.TestOn(100);
            AssertArray.IsGreaterThan(acc, 0.8);
        }
    }
}
