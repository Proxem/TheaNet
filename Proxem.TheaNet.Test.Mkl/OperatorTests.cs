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

namespace Proxem.TheaNet.Test.Mkl
{
    [TestClass]
    public class OperatorTests
    {
        [TestMethod, TestCategory("Simple2")]
        public void TestScalarConst()
        {
            var x = (Scalar<float>)3.0f;
            var f = Op.Function(output: x);
            Assert.AreEqual(f(), 3.0f);
            Assert.AreNotEqual(f(), 4.0f);
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestTensorConst()
        {
            var x = Op.Const(2.0f, 2, 2);
            var f = Op.Function(output: x);
            AssertArray.AreEqual(f(), NN.Const(2.0f, 2, 2));
            AssertArray.AreNotEqual(f(), NN.Const(3.0f, 2, 2));
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestScalarVar()
        {
            var x = Op.Scalar<float>("x");
            var f = Op.Function(input: x, output: x);
            Assert.AreEqual(f(4.0f), 4.0f);
            Assert.AreNotEqual(f(5.0f), 4.0f);
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestScalarShared()
        {
            var x = Op.Shared(4.0f, "x");
            var f = Op.Function(output: x);
            Assert.AreEqual(f(), 4.0f);
            Assert.AreNotEqual(f(), 5.0f);
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestTensorVar()
        {
            var x = Op.Matrix<float>("x");
            var f = Op.Function(input: x, output: x);
            AssertArray.AreEqual(f(NN.Const(3.0f, 2, 2)), NN.Const(3.0f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(3.0f, 2, 2)), NN.Const(4.0f, 2, 2));
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestScalarUnary()
        {
            var x = Op.Scalar<float>("x");
            var f = Op.Function(input: x, output: Op.Abs(x));
            Assert.AreEqual(f(5.0f), 5.0f);
            Assert.AreEqual(f(-5.0f), 5.0f);
            Assert.AreNotEqual(f(6.0f), 5.0f);
            Assert.AreNotEqual(f(-6.0f), 5.0f);
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestTensorUnary()
        {
            var x = Op.Matrix<float>("x");
            var f = Op.Function(input: x, output: Op.Abs(x));
            AssertArray.AreEqual(f(NN.Const(5.0f, 2, 2)), NN.Const(5.0f, 2, 2));
            AssertArray.AreEqual(f(NN.Const(-5.0f, 2, 2)), NN.Const(5.0f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(6.0f, 2, 2)), NN.Const(5.0f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(-6.0f, 2, 2)), NN.Const(5.0f, 2, 2));
        }


        [TestMethod, TestCategory("Simple2")]
        public void TestScalarBinary()
        {
            var x = Op.Scalar<float>("x");
            var y = Op.Scalar<float>("y");
            var f = Op.Function(input: (x, y), output: Op.Max(x, y));
            Assert.AreEqual(f(3, 4), 4);
            Assert.AreEqual(f(3, -4), 3);
            Assert.AreNotEqual(f(3, 4), 3);
            Assert.AreNotEqual(f(3, -4), -4);
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestScalarAdd()
        {
            var x = Op.Scalar<float>("x");
            var y = Op.Scalar<float>("y");
            var f = Op.Function(input: (x, y), output: x + y);
            Assert.AreEqual(f(3, 4), 7);
            Assert.AreEqual(f(3, -4), -1);
            Assert.AreNotEqual(f(3, 4), 3);
            Assert.AreNotEqual(f(3, -4), 4);
        }

        [TestMethod, TestCategory("Simple2"), ExpectedException(typeof(ArgumentException))]
        public void FailScalarAdd()
        {
            var x = Op.Scalar<float>("x");
            var y = Op.Scalar<float>("y");
            var f = Op.Function(input: x, output: x + y);
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestTensorAdd()
        {
            var x = Op.Matrix<float>("x");
            var y = Op.Matrix<float>("y");
            var f = Op.Function(input: (x, y), output: x + y);
            AssertArray.AreEqual(f(NN.Const(3f, 2, 2), NN.Const(4f, 2, 2)), NN.Const(7f, 2, 2));
            AssertArray.AreEqual(f(NN.Const(3f, 2, 2), NN.Const(-4f, 2, 2)), NN.Const(-1f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(3f, 2, 2), NN.Const(4f, 2, 2)), NN.Const(3f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(3f, 2, 2), NN.Const(-4f, 2, 2)), NN.Const(4f, 2, 2));
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestScalarMulAdd()
        {
            var x = Op.Scalar<float>("x");
            var y = Op.Scalar<float>("y");
            var f = Op.Function(input: (x, y), output: 3 * (x + y));
            Assert.AreEqual(f(3, 4), 21);
            Assert.AreEqual(f(3, -4), -3);
            Assert.AreNotEqual(f(3, 4), 7);
            Assert.AreNotEqual(f(3, 4), 9);
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestTensorMulAdd()
        {
            var a = Op.Matrix<float>("a");
            var b = Op.Matrix<float>("b");
            var f = Op.Function(input: (a, b), output: 3 * (a + b));
            AssertArray.AreEqual(f(NN.Const(3f, 2, 2), NN.Const(4f, 2, 2)), NN.Const(21f, 2, 2));
            AssertArray.AreEqual(f(NN.Const(3f, 2, 2), NN.Const(-4f, 2, 2)), NN.Const(-3f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(3f, 2, 2), NN.Const(4f, 2, 2)), NN.Const(7f, 2, 2));
            AssertArray.AreNotEqual(f(NN.Const(3f, 2, 2), NN.Const(4f, 2, 2)), NN.Const(9f, 2, 2));
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestScan()
        {
            var x = Op.Matrix<float>("x");
            var f = Op.Function(input: x, output: Op.Scan(v => v, sequence: x));

            var input = NN.Eye<float>(2);
            var result = f(input);

            AssertArray.AreEqual(input, result);
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestScan2()
        {
            var x = Op.Matrix<float>("x");
            var f = Op.Function(input: x, output: Op.Scan(v => 2f * v, sequence: x));

            var input = NN.Eye<float>(2);
            var result = f(input);

            AssertArray.AreEqual(2 * input, result);
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestScan3()
        {
            var x = Op.Matrix<float>("x");
            var y = Op.Matrix<float>("y");
            var f = Op.Function(input: (x, y), output: Op.Scan((v1, v2) => v1 + v2, sequences: new[] { x, y }));

            var input1 = NN.Eye<float>(2);
            var input2 = 2 * NN.Eye<float>(2);
            var result = f(input1, input2);

            AssertArray.AreEqual(input1 + input2, result);
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestScan4()
        {
            var X = Op.Matrix<float>("X");
            var acc0 = Op.Shared(NN.Zeros<float>(5), "acc0");

            var loop = Op.Scan(fn: (x, acc) => acc + x, sequence: X, outputsInfo: acc0);
            var f = Op.Function(input: X, output: loop[-1]);

            var input1 = NN.Eye<float>(5);
            var result = f(input1);
            AssertArray.AreEqual(new float[] { 1, 1, 1, 1, 1 }, result);
        }

        [TestMethod, TestCategory("Simple2")]
        public void TestSigmoid()
        {
            var x = Op.Vector<float>("x");
            var y = Op.Sigmoid(x);
            var cost = Op.Norm2(y);
            var g = Op.Grad(cost, x);

            var f = Op.Function(input: x, output: (cost, g));
        }
    }
}
