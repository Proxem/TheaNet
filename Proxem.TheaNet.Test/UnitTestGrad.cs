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
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Proxem.NumNet;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    using static XSlicer;

    [TestClass]
    public class UnitTestGrad
    {
        [TestInitialize]
        public void Reset()
        {
            NN.Random.Seed(1234);
            Runtime.Reset();
        }

        [TestMethod]
        public void MaxPassesGradientCheck()
        {
            var x = T.Shared(0f, "x");
            var max = T.Max(x, 0f);

            x.Value = 1;
            AssertTensor.PassesGradientCheck(max, x);

            x.Value = -1;
            AssertTensor.PassesGradientCheck(max, x);
        }

        [TestMethod]
        public void MinPassesGradientCheck()
        {
            var x = T.Shared(0f, "x");
            var min = T.Min(x, 0f);

            x.Value = 1;
            AssertTensor.PassesGradientCheck(min, x);

            x.Value = -1;
            AssertTensor.PassesGradientCheck(min, x);
        }

        [TestMethod]
        /// <see href="http://deeplearning.net/software/theano/tutorial/gradients.html"/>
        public void GradOfX2Is2X()
        {

            var x = T.Scalar<float>("x");
            var y = T.Pow(x, 2);
            var gy = T.Grad(y, x);
            // pp(gy)  # print out the gradient prior to optimization
            // '((fill((x ** 2), 1.0) * 2) * (x ** (2 - 1)))'
            var f = T.Function(x, gy);

            AssertArray.AreAlmostEqual(8.0f, f(4));
            AssertArray.AreAlmostEqual(188.4f, f(94.2f));
        }

        [TestMethod]
        /// <see href="http://deeplearning.net/software/theano/tutorial/gradients.html"/>
        public void TestGradOfNeuralLayer()
        {
            var x = T.Matrix<float>("x");
            var s = T.Sum(1 / (1 + T.Exp(-x)));
            var gs = T.Grad(s, x);
            var dlogistic = T.Function(x, gs);
            var m = NN.Array(new float[,] {
                { 0, 1 },
                { -1, -2 }
            });

            var expected = NN.Array(new float[,] {
                { 0.25f, 0.196612f },
                { 0.196612f, 0.10499359f }
            });

            AssertArray.AreAlmostEqual(expected, dlogistic(m));
        }

        [TestMethod]
        public void GradAndFiniteDifferenceAgreeOnPerceptron()
        {
            var x = T.Vector<float>("x");
            int n = 20, m = 5;
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, m, n).As<float>(), "W");
            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, m, n).As<float>(), "W_op");

            var x_ = NN.Random.Uniform(-1f, 1f, n).As<float>();
            var loss = T.Norm2(T.Dot(W, x) - T.Dot(W_op, x));

            var gradW = T.Function(x, T.Grad(loss, W))(x_);
            var dW = T.FiniteDifference_(new[] { x }, loss, W);
            var finiteW = NN.Zeros(m, n);

            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    finiteW.Item[i, j] = dW(x_, i, j, 0.001f);

            AssertArray.AreAlmostEqual(finiteW, gradW, relativeErr: 1e-3f, absErr: 1e-3f);
        }

        [TestMethod]
        public void PerceptronPassesGradientCheck()
        {
            var x = T.Vector<float>("x");
            int n = 20, m = 5;
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, m, n).As<float>(), "W");
            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, m, n).As<float>(), "W_op");

            var loss = T.Norm2(T.Dot(W, x) - T.Dot(W_op, x));
            AssertTensor.PassesGradientCheck(x, loss, W);
        }

        [TestMethod]
        public void TanhPerceptronPassesGradientCheck()
        {
            var x = T.Vector<float>("x");
            int n = 20, m = 5;
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, m, n).As<float>(), "W");
            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, m, n).As<float>(), "W_op");
            var loss = T.Norm2(T.Tanh(T.Dot(W, x)) - T.Dot(W_op, x));

            AssertTensor.PassesGradientCheck(x, loss, W);
        }

        [TestMethod]
        public void SoftmaxPerceptronPassesGradientCheck()
        {
            int n = 20, m = 5;
            var x = T.Vector<float>("x");
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, m, n).As<float>(), "W");
            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, m, n).As<float>(), "W_op");

            var loss = T.Norm2(T.Softmax(T.Dot(W, x)) - T.Dot(W_op, x));
            AssertTensor.PassesGradientCheck(x, loss, W);
        }

        [TestMethod]
        public void GradAndFiniteDifferenceAgreeOnSoftmaxPerceptron()
        {
            var x = T.Vector<float>("x");
            int n = 20, m = 5;
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, m, n).As<float>(), "W");
            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, m, n).As<float>(), "W_op");

            var x_ = NN.Random.Uniform(-1f, 1f, n).As<float>();
            var loss = T.Norm2(T.Softmax(T.Dot(W, x)) - T.Dot(W_op, x));

            var gradW = T.Function(x, T.Grad(loss, W))(x_);
            var dW = T.FiniteDifference_(new[] { x }, loss, W);
            var finiteW = NN.Zeros(m, n);

            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    finiteW.Item[i, j] = dW(x_, i, j, 0.01f);

            AssertArray.AreAlmostEqual(finiteW, gradW, relativeErr: 1e-2f);
        }

        [TestMethod]
        public void TanhPerceptronWithBiasPassesGradientCheck()
        {
            var x = T.Vector<float>("x");
            int n = 20, m = 5;
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, m, n).As<float>(), "W");
            var b1 = T.Shared(NN.Random.Uniform(-1f, 1f, n).As<float>(), "b1");
            var b2 = T.Shared(NN.Random.Uniform(-1f, 1f, m).As<float>(), "b2");
            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, m, n).As<float>(), "W_op");

            var loss = T.Norm2(T.Tanh(T.Dot(W, x + b1)) + b2 - T.Dot(W_op, x));

            AssertTensor.PassesGradientCheck(x, loss, b1);
            AssertTensor.PassesGradientCheck(x, loss, b2);
            AssertTensor.PassesGradientCheck(x, loss, W);
        }

        [TestMethod]
        public void TanhNNPassesGradientCheck()
        {
            var x0 = T.Vector<float>("x");

            int n0 = 20, n1 = 5, n2 = 20;
            var W1 = T.Shared(NN.Random.Uniform(-1f, 1f, n1, n0).As<float>(), "W1");
            var W2 = T.Shared(NN.Random.Uniform(-1f, 1f, n2, n1).As<float>(), "W2");
            var b1 = T.Shared(NN.Random.Uniform(-1f, 1f, n1).As<float>(), "b1");
            var b2 = T.Shared(NN.Random.Uniform(-1f, 1f, n2).As<float>(), "b2");

            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, n2, n0).As<float>(), "W_op");

            var x1 = T.Tanh(T.Dot(W1, x0) + b1);
            var x2 = T.Tanh(T.Dot(W2, x1) + b2);
            var loss = T.Norm2(x2 - T.Dot(W_op, x0));

            AssertTensor.PassesGradientCheck(x0, loss, b1);
            AssertTensor.PassesGradientCheck(x0, loss, b2);
            AssertTensor.PassesGradientCheck(x0, loss, W1);
            AssertTensor.PassesGradientCheck(x0, loss, W2);
        }

        [TestMethod]
        public void ReluNNPassesGradientCheck()
        {
            var x0 = T.Vector<float>("x");

            int n0 = 20, n1 = 5, n2 = 20;
            var W1 = T.Shared(NN.Random.Uniform(-1f, 1f, n1, n0).As<float>(), "W1");
            var W2 = T.Shared(NN.Random.Uniform(-1f, 1f, n2, n1).As<float>(), "W2");
            var b1 = T.Shared(NN.Random.Uniform(-1f, 1f, n1).As<float>(), "b1");
            var b2 = T.Shared(NN.Random.Uniform(-0.1f, 0.1f, n2).As<float>(), "b2");

            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, n2, n0).As<float>(), "W_op");

            var x1 = T.ReLu(T.Dot(W1, x0) + b1);
            var x2 = T.ReLu(T.Dot(W2, x1) + b2);
            var loss = T.Norm2(x2 - T.Dot(W_op, x0));

            AssertTensor.PassesGradientCheck(x0, loss, b1);
            AssertTensor.PassesGradientCheck(x0, loss, W1);
            AssertTensor.PassesGradientCheck(x0, loss, W2);
            AssertTensor.PassesGradientCheck(x0, loss, b2);
        }

        [TestMethod]
        public void TanhNNAsEinsteinSumPassesGradientCheck()
        {
            var x0 = T.Vector<float>("x");

            int n0 = 20, n1 = 5, n2 = 20;
            var W1 = T.Shared(NN.Random.Uniform(-1f, 1f, n1, n0).As<float>(), "W1");
            var W2 = T.Shared(NN.Random.Uniform(-1f, 1f, n2, n1).As<float>(), "W2");
            var b1 = T.Shared(NN.Random.Uniform(-1f, 1f, n1).As<float>(), "b1");
            var b2 = T.Shared(NN.Random.Uniform(-0.1f, 0.1f, n2).As<float>(), "b2");

            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, n2, n0).As<float>(), "W_op");

            var x1 = T.Tanh(T.EinsteinSum(W1, x0, "ij,j->i") + b1);
            var x2 = T.Tanh(T.EinsteinSum(W2, x1, "ij,j->i") + b2);
            var loss = T.Norm2(x2 - T.Dot(W_op, x0));

            AssertTensor.PassesGradientCheck(x0, loss, b1);
            AssertTensor.PassesGradientCheck(x0, loss, W1);
            AssertTensor.PassesGradientCheck(x0, loss, W2);
            AssertTensor.PassesGradientCheck(x0, loss, b2);
        }

        [TestMethod]
        public void TanhNNTransposedAsEinsteinSumPassesGradientCheck()
        {
            var x0 = T.Vector<float>("x");

            int n0 = 20, n1 = 5, n2 = 20;
            var W1 = T.Shared(NN.Random.Uniform(-1f, 1f, n0, n1).As<float>(), "W1");
            var W2 = T.Shared(NN.Random.Uniform(-1f, 1f, n1, n2).As<float>(), "W2");
            var b1 = T.Shared(NN.Random.Uniform(-1f, 1f, n1).As<float>(), "b1");
            var b2 = T.Shared(NN.Random.Uniform(-0.1f, 0.1f, n2).As<float>(), "b2");

            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, n0, n2).As<float>(), "W_op");

            var x1 = T.Tanh(T.EinsteinSum(x0, W1, "i,ij->j") + b1);
            var x2 = T.Tanh(T.EinsteinSum(x1, W2, "i,ij->j") + b2);
            var loss = T.Norm2(x2 - T.Dot(W_op, x0));

            AssertTensor.PassesGradientCheck(x0, loss, b1);
            AssertTensor.PassesGradientCheck(x0, loss, W1);
            AssertTensor.PassesGradientCheck(x0, loss, W2);
            AssertTensor.PassesGradientCheck(x0, loss, b2, absErr: 0.002f);
        }

        [TestMethod]
        public void TensorDot3Dx1DPassesGradientCheck()
        {
            var x = T.Vector<float>("x");
            int n = 6, m = 4, l = 2;
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, l, m, n).As<float>(), "W");
            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, l, m, n).As<float>(), "W_op");

            var loss = T.Norm2(T.Dot(W, x) - T.Dot(W_op, x));
            AssertTensor.PassesGradientCheck(x, loss, W, relativeErr: 1e-3f, absErr: 1e-4f);
        }

        [TestMethod]
        public void TensorDot3Dx1DAsEinsteinPassesGradientCheck()
        {
            var x = T.Vector<float>("x");
            int n = 6, m = 4, l = 2;
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, l, m, n).As<float>(), "W");
            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, l, m, n).As<float>(), "W_op");

            var loss = T.Norm2(T.EinsteinSum(W, x, "lmn,n->lm") - T.Dot(W_op, x));
            AssertTensor.PassesGradientCheck(x, loss, W);
        }

        [TestMethod]
        public void TensorDot3Dx2DPassesGradientCheck()
        {
            int n = 6, m = 4, l = 2;
            var x = T.Matrix<float>("x");
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, l, m, n).As<float>(), "W");
            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, l, m, n).As<float>(), "W_op");

            var loss = T.Norm2(T.Dot(W, x) - T.Dot(W_op, x));
            AssertTensor.PassesGradientCheck(x, loss, W);
        }

        [TestMethod]
        public void TensorDot2Dx3DPassesGradientCheck()
        {
            int n = 6, m = 4, l = 2, k = 3;
            var x = T.Tensor3<float>(n, m, l, "x");
            var xT = x.DimShuffle(1, 0, 2);
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, l, k).As<float>(), "W");
            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, l, k).As<float>(), "W_op");

            var loss = T.Norm2(T.Softmax(T.Dot(xT, W)) - T.Dot(xT, W_op));
            AssertTensor.PassesGradientCheck(x, loss, W);
        }

        [TestMethod]
        public void TensorDot3Dx3DPassesGradientCheck()
        {
            int n = 6, m = 4, l = 2, k = 5;
            var x = T.Tensor3<float>(k, l, m, "x");
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, l, m, n).As<float>(), "W");
            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, l, m, n).As<float>(), "W_op");

            var axesW = new[] { 0, 1 };
            var axesX = new[] { 1, 2 };
            var loss = T.Norm2(T.TensorDot(W, axesW, x, axesX) - T.TensorDot(W_op, axesW, x, axesX));
            AssertTensor.PassesGradientCheck(x, loss, W);
        }

        [TestMethod]
        public void TensorDot3Dx2DAsEinsteinPassesGradientCheck()
        {
            int n = 6, m = 4, l = 2;
            var x = T.Matrix<float>("x");
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, l, m, n).As<float>(), "W");
            var W_op = T.Shared(NN.Random.Uniform(-1f, 1f, l, m, n).As<float>(), "W_op");

            var loss = T.Norm2(T.EinsteinSum(W, x, "lmn,nx->lmx") - T.Dot(W_op, x));
            AssertTensor.PassesGradientCheck(x, loss, W);
        }

        [TestMethod]
        public void TensorDot3Dx2DAsEinsteinMatchesTensorDot()
        {
            int n = 6, m = 4, l = 2;
            var x = T.Matrix<float>("x");
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, l, m, n).As<float>(), "W");

            var loss = T.Norm2(T.EinsteinSum(W, x, "lmn,nx->lmx") - T.Dot(W, x));
            var err = T.Function(input: x, output: loss);

            for(int i = 0; i < 10; ++i)
                AssertArray.IsLessThan(err(NN.Random.Uniform(-1f, 1f, n, 10)), 1e-5);
        }

        [TestMethod, TestCategory("Slow")]
        public void Convolve2DPassesGradientCheck()
        {
            //int[] poolingShape = new int[] { 1, 1 };
            int[] kernelShape = new int[] { 7, 7 };
            int[] inputShape = new int[] { 100, 100 };
            var iS = NN.Array(inputShape).As<float>();
            var kS = NN.Array(kernelShape).As<float>();
            // layers
            var W = T.Shared(NN.Random.Uniform(-0.01f, 0.01f, kernelShape).As<float>(), "W");
            //var flatShape = ((inputShape[0] + kernelShape[0] - 1) / poolingShape[0] )  * ((inputShape[1] + kernelShape[1] - 1) / poolingShape[1] );
            var flatShape = ((inputShape[0] + kernelShape[0] - 1)) * ((inputShape[1] + kernelShape[1] - 1));
            var scaling = (((iS[0] + kS[0] - 1f)) + ((iS[1] + kS[1] - 1f)));
            var S = T.Shared(NN.Random.Uniform(-10f, 10f, 2, flatShape).As<float>()/scaling, "S");
            var Sb = T.Shared(NN.Zeros<float>(2, 1), "Sb");

            var x = T.Matrix<float>(inputShape[0], inputShape[1], "x");  // [inputLength]
            var h = T.Sigmoid(T.Convolve2d(x, W, mode: ConvMode.Full));
            //h = T.MaxPooling2d(h, poolingShape[0], poolingShape[1], true);
            h = h.Reshape(flatShape, 1);
            var debug = (T.Dot(S, h) + Sb).Reshape(2);
            var pred = T.Softmax(debug);
            var nll = -T.Mean(T.Log(pred)[1]);

            AssertTensor.PassesGradientCheck(x, nll, W, relativeErr: 1e-3f, absErr: 1e-3f);
        }

        [TestMethod]
        public void ConcatPassesGradientCheck()
        {
            var x = T.Shared(NN.Random.Uniform(-1f, 1f, 4, 10), "x");
            var y = T.Shared(NN.Random.Uniform(-1f, 1f, 6, 10), "y");

            var z = T.Concat(0, x, y);
            var loss = T.Norm2(z[Range(2, 8)]);

            AssertTensor.PassesGradientCheck(loss, x);
            AssertTensor.PassesGradientCheck(loss, y);
        }

        [TestMethod]
        public void SlicingCompiles()
        {
            var x = T.Matrix<float>(10, 5, "x");
            var loss = T.Norm2(x[From(2)]);

            var f = T.Function(input: x, output: loss);
            f(NN.Range<float>(50).Reshape(10, 5));
        }

        [TestMethod]
        public void MaxPoolingPassesGradientCheck()
        {
            var x_ = NN.Array(new float[,]
            {
                { 0, 1, 0 },
                { 1, 0, 0 },
                { 0, 0, 1 },
            });

            var x = T.Shared(x_, "x");
            var x_pooled = T.Max(x, axis: 1);
            var loss = T.Sum(x_pooled);

            AssertTensor.PassesGradientCheck(loss, x);
        }
    }
}
