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

using static Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestSimple
    {
        [TestMethod, TestCategory("Simple")]
        public void TestScalarConst()
        {
            var x = (Scalar<float>)3.0f;
            var f = Function(output: x);
            Assert.AreEqual(f(), 3.0f);
            Assert.AreNotEqual(f(), 4.0f);
        }

        [TestMethod, TestCategory("Simple")]
        public void TestTensorConst()
        {
            var x = Const(2.0f, 2, 2);
            var f = Function(output: x);
            AssertArray.AreEqual(f(), NN.Const(2.0f, 2, 2));
            AssertArray.AreNotEqual(f(), NN.Const(3.0f, 2, 2));
        }

        [TestMethod, TestCategory("Simple")]
        public void TestScalarVar()
        {
            var x = Scalar<float>("x");
            var f = Function(input: x, output: x);
            Assert.AreEqual(f(4.0f), 4.0f);
            Assert.AreNotEqual(f(5.0f), 4.0f);
        }

        [TestMethod, TestCategory("Simple")]
        public void TestScalarShared()
        {
            var x = Shared(4.0f, "x");
            var f = Function(output: x);
            Assert.AreEqual(f(), 4.0f);
            Assert.AreNotEqual(f(), 5.0f);
        }

        [TestMethod, TestCategory("Simple")]
        public void TestTensorVar()
        {
            var x = Matrix<float>("x");
            var f = Function(input: x, output: x);
            AssertArray.AreEqual(f(NN.Const(3.0f, 2, 2)), NN.Const(3.0f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(3.0f, 2, 2)), NN.Const(4.0f, 2, 2));
        }

        [TestMethod, TestCategory("Simple")]
        public void TestScalarUnary()
        {
            var x = Scalar<float>("x");
            var f = Function(input: x, output: Abs(x));
            Assert.AreEqual(f(5.0f), 5.0f);
            Assert.AreEqual(f(-5.0f), 5.0f);
            Assert.AreNotEqual(f(6.0f), 5.0f);
            Assert.AreNotEqual(f(-6.0f), 5.0f);
        }

        [TestMethod, TestCategory("Simple")]
        public void TestTensorUnary()
        {
            var x = Matrix<float>("x");
            var f = Function(input: x, output: Abs(x));
            AssertArray.AreEqual(f(NN.Const(5.0f, 2, 2)), NN.Const(5.0f, 2, 2));
            AssertArray.AreEqual(f(NN.Const(-5.0f, 2, 2)), NN.Const(5.0f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(6.0f, 2, 2)), NN.Const(5.0f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(-6.0f, 2, 2)), NN.Const(5.0f, 2, 2));
        }


        [TestMethod, TestCategory("Simple")]
        public void TestScalarBinary()
        {
            var x = Scalar<float>("x");
            var y = Scalar<float>("y");
            var f = Function(input1: x, input2: y, output: Max(x, y));
            Assert.AreEqual(f(3, 4), 4);
            Assert.AreEqual(f(3, -4), 3);
            Assert.AreNotEqual(f(3, 4), 3);
            Assert.AreNotEqual(f(3, -4), -4);
        }

        [TestMethod, TestCategory("Simple")]
        public void TestScalarAdd()
        {
            var x = Scalar<float>("x");
            var y = Scalar<float>("y");
            var f = Function(input1: x, input2: y, output: x + y);
            Assert.AreEqual(f(3, 4), 7);
            Assert.AreEqual(f(3, -4), -1);
            Assert.AreNotEqual(f(3, 4), 3);
            Assert.AreNotEqual(f(3, -4), 4);
        }

        [TestMethod, TestCategory("Simple"), ExpectedException(typeof(ArgumentException))]
        public void FailScalarAdd()
        {
            var x = Scalar<float>("x");
            var y = Scalar<float>("y");
            var f = Function(input: x, output: x + y);
        }

        [TestMethod, TestCategory("Simple")]
        public void TestTensorAdd()
        {
            var x = Matrix<float>("x");
            var y = Matrix<float>("y");
            var f = Function(input1: x, input2: y, output: x + y);
            AssertArray.AreEqual(f(NN.Const(3f, 2, 2), NN.Const(4f, 2, 2)), NN.Const(7f, 2, 2));
            AssertArray.AreEqual(f(NN.Const(3f, 2, 2), NN.Const(-4f, 2, 2)), NN.Const(-1f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(3f, 2, 2), NN.Const(4f, 2, 2)), NN.Const(3f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(3f, 2, 2), NN.Const(-4f, 2, 2)), NN.Const(4f, 2, 2));
        }

        [TestMethod, TestCategory("Simple")]
        public void TestScalarMulAdd()
        {
            var x = Scalar<float>("x");
            var y = Scalar<float>("y");
            var f = Function(input1: x, input2: y, output: 3 * (x + y));
            Assert.AreEqual(f(3, 4), 21);
            Assert.AreEqual(f(3, -4), -3);
            Assert.AreNotEqual(f(3, 4), 7);
            Assert.AreNotEqual(f(3, 4), 9);
        }

        [TestMethod, TestCategory("Simple")]
        public void TestTensorMulAdd()
        {
            var a = Matrix<float>("a");
            var b = Matrix<float>("b");
            var f = Function(input1: a, input2: b, output: 3 * (a + b));
            AssertArray.AreEqual(f(NN.Const(3f, 2, 2), NN.Const(4f, 2, 2)), NN.Const(21f, 2, 2));
            AssertArray.AreEqual(f(NN.Const(3f, 2, 2), NN.Const(-4f, 2, 2)), NN.Const(-3f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(3f, 2, 2), NN.Const(4f, 2, 2)), NN.Const(7f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(3f, 2, 2), NN.Const(4f, 2, 2)), NN.Const(9f, 2, 2));
        }

        [TestMethod, TestCategory("Simple")]
        public void TestScan()
        {
            var x = Matrix<float>("x");
            var f = Function(input: x, output: Scan(v => v, sequence: x));

            var input = NN.Eye<float>(2);
            var result = f(input);

            AssertArray.AreEqual(input, result);
        }

        [TestMethod, TestCategory("Simple")]
        public void TestScan2()
        {
            var x = Matrix<float>("x");
            var f = Function(input: x, output: Scan(v => 2f * v, sequence: x));

            var input = NN.Eye<float>(2);
            var result = f(input);

            AssertArray.AreEqual(2 * input, result);
        }

        [TestMethod, TestCategory("Simple")]
        public void TestScan3()
        {
            var x = Matrix<float>("x");
            var y = Matrix<float>("y");
            var f = Function(input1: x, input2: y, output: Scan((v1, v2) => v1 + v2, sequences: new[] { x, y }));

            var input1 = NN.Eye<float>(2);
            var input2 = 2 * NN.Eye<float>(2);
            var result = f(input1, input2);

            AssertArray.AreEqual(input1 + input2, result);
        }

        [TestMethod, TestCategory("Simple")]
        public void ScanWorksOnAxis0()
        {
            var X = Matrix<float>("X");
            var acc0 = Zeros<float>(X.Shape[1]).Named("acc0");

            var loop = Scan(fn: (x, acc) => acc + x, sequence: X, outputsInfo: acc0, axis: 0);
            loop.AssertOfShape(X);
            var f = Function(input: X, output: loop[-1]);

            var X_ = NN.Random.Uniform(-1f, 1f, 10, 20);
            var result = f(X_);
            AssertArray.AreEqual(X_.Sum(axis: 0), result);
        }

        [TestMethod, TestCategory("Simple")]
        public void ScanWorksOnAxis1()
        {
            var X = Matrix<float>("X");
            var acc0 = Zeros<float>(X.Shape[0]).Named("acc0");

            var loop = Scan(fn: (x, acc) => acc + x, sequence: X, outputsInfo: acc0, axis: 1);
            Assert.IsTrue(loop.Shape.WillEqualTo(X.DimShuffle(1, 0).Shape));
            var f = Function(input: X, output: loop[-1]);

            var X_ = NN.Random.Uniform(-1f, 1f, 10, 20);
            var result = f(X_);
            AssertArray.AreEqual(X_.Sum(axis: 1), result);
        }


        [TestMethod, TestCategory("Simple")]
        public void TestSigmoid()
        {
            var x = Vector<float>("x");
            var y = Sigmoid(x);

            var f = Function(input: x, output: y);
        }

        [TestMethod, TestCategory("Simple")]
        public void TestSigmoidGrad()
        {
            var x = Vector<float>("x");
            var sigmoid = Sigmoid(x).Named("sigmoid");
            var cost = (0.5f * Norm2(sigmoid)).Named("cost");
            var g = Grad(cost, x).Named("g");

            var f = Function(input: x, output: g);
            f(NN.Range<float>(5));
        }

        [TestMethod, TestCategory("Simple")]
        public void TestSigmoidCostGrad()
        {
            var x = Vector<float>("x");
            var sigmoid = Sigmoid(x).Named("sigmoid");
            var cost = (0.5f * Norm2(sigmoid)).Named("cost");
            var g = Grad(cost, x).Named("g");

            var f = Function(input: x, output1: cost, output2: g);
            f(NN.Range<float>(5));
        }

        [TestMethod, TestCategory("Simple")]
        public void CanNameConst()
        {
            var pi = Const((float)Math.PI).Named("pi");
            var f = Function(pi);
            f();
        }
    }
}
