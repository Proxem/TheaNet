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
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Proxem.NumNet;
using Proxem.NumNet.Single;
using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestFloatExpr
    {
        [TestInitialize]
        public void Initialize()
        {
            Runtime.Reset();
        }

        [TestMethod]
        public void TestExp()
        {
            var x = T.Scalar<float>("x");
            var e = T.Exp(x / 5);
            var de = T.Grad(e, x);

            var f = T.Function(x, e);
            Assert.AreEqual((float)Math.Exp(4f / 5f), f(4));
            Assert.AreEqual((float)Math.Exp(5f / 5f), f(5));

            var df = T.Function(x, de);
            Assert.AreEqual((float)Math.Exp(4f / 5f) / 5f, df(4));
            Assert.AreEqual((float)Math.Exp(5f / 5f) / 5f, df(5));

            var fdf = T.Function(x, new[] { e, de });
            var res = fdf(4);
            Assert.AreEqual(f(4), res[0]);
            Assert.AreEqual(df(4), res[1]);

            res = fdf(5);
            Assert.AreEqual(f(5), res[0]);
            Assert.AreEqual(df(5), res[1]);
        }

        [TestMethod]
        public void TestTanh()
        {
            var x = T.Scalar<float>("x");
            var e = T.Tanh(x / 5);

            var f = T.Function(x, e);
            Assert.AreEqual((float)Math.Tanh(4f / 5f), f(4));
            Assert.AreEqual((float)Math.Tanh(5f / 5f), f(5));

            var df = T.Function(x, T.Grad(e, x));

            Func<float, float> dtanh = y => 1 - (float)Math.Tanh(y) * (float)Math.Tanh(y);
            Assert.AreEqual(dtanh(4f / 5f) / 5f, df(4));
            Assert.AreEqual(dtanh(5f / 5f) / 5f, df(5));
        }

        [TestMethod]
        public void TestTwoVarExpr()
        {
            var x = T.Scalar<float>("x");
            var y = T.Scalar<float>("y");
            var e = T.Tanh(x / 5) * T.Exp(0.5f * y);
            var de_dx = T.Grad(e, x);

            var f = T.Function(input: (x, y), output: e);
            Assert.AreEqual((float)Math.Tanh(4f / 5f) * (float)Math.Exp(0.5f * 3f), f(4, 3));
            Assert.AreEqual((float)Math.Tanh(5f / 5f) * (float)Math.Exp(0.5f * 4f), f(5, 4));

            var df_dx = T.Function(input: (x, y), output: de_dx);
            Func<float, float> dtanh = z => 1 - (float)Math.Tanh(z) * (float)Math.Tanh(z);
            AssertArray.AreAlmostEqual(dtanh(4f / 5f) / 5f * (float)Math.Exp(0.5f * 3f), df_dx(4f, 3f));
            AssertArray.AreAlmostEqual(dtanh(5f / 5f) / 5f * (float)Math.Exp(0.5f * 4f), df_dx(5f, 4f));

            var df_dy = T.Function(input: (x, y), output: T.Grad(e, y));
            AssertArray.AreAlmostEqual((float)Math.Tanh(4f / 5f) * 0.5f * (float)Math.Exp(0.5f * 3f), df_dy(4f, 3f));
            AssertArray.AreAlmostEqual((float)Math.Tanh(5f / 5f) * 0.5f * (float)Math.Exp(0.5f * 4f), df_dy(5f, 4f));
        }

        [TestMethod, ExpectedException(typeof(ArgumentException)), TestCategory("Exception")]
        public void FailTwoVar()
        {
            var x = T.Scalar<float>("x");
            var y = T.Scalar<float>("y");
            var f = T.Function(x, x + y);
        }

        [TestMethod, ExpectedException(typeof(ArgumentException)), TestCategory("Exception")]
        public void FailTwoVarExprNotShared()
        {
            var x = T.Scalar<float>("x");
            var y = T.Scalar<float>("y");
            var e = T.Tanh(x / 5) * T.Exp(0.5f * y);

            // When not all arguments of a function are precised,
            // the 'Function' throws an exception
            var g = T.Function(x, e);
            var d = g(3f);
        }


        [TestMethod]
        public void TestTwoVarExprShared()
        {
            var x = T.Scalar<float>("x");
            var y = T.Shared(0f, "y");
            var e = T.Tanh(x / 5) * T.Exp(0.5f * y);

            // For shared variables, no exception
            var g = T.Function(x, e);
            Assert.AreEqual(g(3f), (float)Math.Tanh(3f / 5));

            y.Value = 2;
            Assert.AreEqual(g(3f), (float)Math.Tanh(3f / 5) * (float)Math.Exp(1));
        }

        [TestMethod, ExpectedException(typeof(ArgumentException)), TestCategory("Exception")]
        public void FailMissingVariable()
        {
            var x = T.Matrix<float>("x");
            var y = T.Matrix<float>("y");
            var z = x + y;
            var f = T.Function(x, z);   // "y" is missing

            f(NN.Array(1f, 2f, 3f));  // should throw exception
        }

        [TestMethod]
        public void TestShared()
        {
            var x = T.Vector<float>("x");
            var y = T.Shared(NN.Array<float>(2, 5, 8), "y");
            var z = x + y;
            var f = T.Function(x, z);

            var result = f(NN.Array<float>(1, 2, 3));
            AssertArray.AreAlmostEqual(NN.Array<float>(3, 7, 11), result);

            y.Value = NN.Array<float>(1, 1, 1);
            var result2 = f(NN.Array<float>(1, 2, 3));
            AssertArray.AreAlmostEqual(NN.Array<float>(2, 3, 4), result2);
        }

        [TestMethod]
        public void TestLogOfSoftmax()
        {
            var x = T.Vector<float>("x");
            var y = T.Scalar<int>("y");
            var W = T.Shared(NN.Range<float>(5 * 4).Reshape(5, 4), "W");
            var output = T.Tanh(T.Dot(W, x));
            var p_y_given_x = T.Softmax(output);
            var y_pred = T.Sum(T.Argmax(p_y_given_x, axis: 0));

            var nll = -T.Log(T.Sum(p_y_given_x[y]));
            var loss = T.Function(input: (x, y), output: nll);
        }

        [TestMethod]
        public void LogSumExpPassesGradientCheck()
        {
            var x = T.Matrix<float>(5, 4, "x");
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, 8, 5).As<float>(), "W");

            AssertTensor.PassesGradientCheck(x, T.Sum(T.LogSumExp(T.Dot(W, x))), W);
            AssertTensor.PassesGradientCheck(x, T.LogSumExp(T.Dot(W, x)).Item[2], W);

            var y = T.Vector<float>(10, "y");
            var b = T.Shared(NN.Random.Uniform(-1f, 1f, 10).As<float>(), "b");

            AssertTensor.PassesGradientCheck(y, (Scalar<float>)T.LogSumExp(y + b), b);
        }

        [TestMethod]
        public void ItemPassesGradientCheck()
        {
            var y = T.Vector<float>(10, "y");
            var b = T.Shared(NN.Random.Uniform(-1f, 1f, 10).As<float>(), "b");

            AssertTensor.PassesGradientCheck(y, (y + b).Item[5], b);
            AssertTensor.PassesGradientCheck(y, (y + b).Item[-3], b);
        }

        [TestMethod]
        [TestCategory("TODO")]
        // TODO: this test doesn't work due to bugs in Elementwise
        public void PushCoherentGradientOnSimpleAbstraction()
        {
            var x = T.Shared(NN.Range<float>(4), "x");
            var b = T.Scalar<float>("b");

            var y = T.Apply(x, x_ => x_ + b);
            var loss = T.Sum(y);

            var dL_db = T.Function(b, T.Grad(loss, b));
            //dL_db should be 4;

            AssertArray.AreEqual(NN.Const(4f, 10), NN.Range<float>(10).Apply(dL_db));
        }

        [TestMethod]
        [TestCategory("TODO")]
        // TODO: this test doesn't work due to bugs in Elementwise
        public void PushCoherentGradientOnComplexAbstraction()
        {
            var x = T.Shared(NN.Range<float>(4), "x");
            var b = T.Scalar<float>("b");

            var y = T.Apply(x, x_ => (x_ > 0f) * b + x_ + b);
            var loss = T.Sum(y);

            AssertArray.WithMessage("Can't compile the gradient.", () =>
                T.Function(input: b, output: T.Grad(loss, b))
            );
            AssertTensor.PassesGradientCheck(loss, b);
        }

        [TestMethod]
        [ExpectedException(typeof(RankException)), TestCategory("Exception")]
        public void FailForChekShapes()
        {
            var a0 = T.Matrix<float>(1, 10, "a0");
            var xs = T.Tensor3<float>(20, 10, 10, "xs");
            var sum = T.Scan((x, a) => x + a, xs, a0);
        }

        [TestMethod]
        public void TestLt()
        {
            var x = T.Matrix<float>("x_lt");
            var x_ = NN.Array(new float[,] {
                {0, 3, 7},
                {5, 2, 0}
            });

            var y = T.Shared(NN.Array(new float[,] {
                {1, 2, 5},
                {4, 2, 3}
            }), "y_lt");

            var lt = T.Function(input: x, output: x < y);

            var expected = NN.Array(new float[,]{
                {1, 0, 0},
                {0, 0, 1}
            });

            AssertArray.AreEqual(expected, lt(x_));
        }

        [TestMethod]
        public void TestLtEq()
        {
            var x = T.Matrix<float>("x");
            var x_ = NN.Array(new float[,] {
                {0, 3, 7},
                {5, 2, 0}
            });

            var y = T.Shared(NN.Array(new float[,] {
                {1, 2, 5},
                {4, 2, 3}
            }), "y");

            var lt = T.Function(input: x, output: x <= y);

            var expected = NN.Array(new float[,]{
                {1, 0, 0},
                {0, 1, 1}
            });

            AssertArray.AreEqual(expected, lt(x_));
        }

        [TestMethod]
        public void TestOnehotDotM()
        {
            var M = T.Matrix<float>("M");
            var X = T.Matrix<float>("X");
            var a = T.Vector<float>("a");
            var oneHot = T.OneHot(X.Shape, 1, a);
            var B = T.Dot(oneHot, M);

            var M_ = NN.Array(new float[,] {
                {0, 3, 7},
                {5, 2, 0}
            });
            var X_ = NN.Zeros(4, 2);
            var a_ = NN.Array<float>(1, -1);

            var B_ = Op.Function(input: (M, X, a), output: B);
            var B_pred = B_(M_, X_, a_);

            var Y_ = X_.Copy();
            Y_[1] = a_;
            var B_exp = Y_.Dot(M_);

            AssertArray.AreEqual(B_exp, B_pred);
        }

        [TestMethod]
        public void TestGradientThroughImplicitBroadcast()
        {
            var M = T.Matrix<float>("M");
            var W_ = NN.Array(new float[,] {
                {0, 0, 1},
                {1, 1, 0}
            });
            var W = T.Shared(W_, "W");
            var X = T.Matrix<float>(-1, 1, "X");
            var loss = T.Sum((M + X) * W);

            var M_ = NN.Array(new float[,] {
                {0, 3, 7},
                {5, 2, 0}
            });
            var X_ = NN.Ones(2).Reshape(-1, 1);

            var dL = T.Function(input: (X, M), output: T.Grad(loss, X));
            var dX = NN.Array(new float[,] { { 1 }, { 2 } });
            AssertArray.AreAlmostEqual(dX, dL(X_, M_));
        }

        [TestMethod]
        public void TestTensorDot()
        {
            var V_ = NN.Array(new float[,,] {
                { {0, 0,  1}, {1, 1,  0} },
                { {1, 0, -1}, {2, 0, -1} },
            });
            var V = T.Shared(V_, "W");
            var X = T.Matrix<float>(-1, 2, "X");
            var loss = T.Norm2(T.Dot(V, X));

            var X_ = NN.Random.Uniform(-1f, 1f, 3, 2);

            var dLdV = T.Function(input: X, output: (loss, T.Grad(loss, V)));
            dLdV(X_);
        }

        [TestMethod]
        public void DimShufflePassesGradientCheck()
        {
            var X = T.Matrix<float>(5, 3, "X");
            var b = T.Shared(NN.Random.Uniform(-1f, 1f, 3), "b");

            var b2 = b.DimShuffle('x', 0);
            var Xb = X * b2;
            var loss = T.Norm2(Xb);
            AssertTensor.PassesGradientCheck(X, loss, b);
        }

        [TestMethod]
        public void IndexingPassesGradientCheck()
        {
            var maxLength = 20;
            var n = 10;
            var ids = T.Vector<int>("X");
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, n, n), "W");

            var loss = T.Sum(W[ids]);
            AssertTensor.PassesGradientCheck(
                ids, loss, W,
                init: () => NN.Random.Uniform<int>(0, n - 1, (int)(NN.Random.NextDouble() * maxLength) + 1)
            );

            var loss2 = T.Sum(W[ids, ids]);
            AssertTensor.PassesGradientCheck(
                ids, loss2, W,
                init: () => NN.Random.Uniform<int>(0, n - 1, (int)(NN.Random.NextDouble() * maxLength) + 1)
            );
        }
    }
}
