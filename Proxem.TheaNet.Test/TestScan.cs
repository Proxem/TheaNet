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
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Proxem.NumNet;
using Proxem.NumNet.Single;
using Proxem.TheaNet;
using Proxem.TheaNet.Operators.FloatTensors;
using Proxem.TheaNet.Operators.Tensors;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestScan
    {
        [TestInitialize]
        public void ResetRuntime()
        {
            Runtime.Reset();
        }

        [TestMethod]
        public void TestLoop1()
        {
            // Computing tanh(x(t).dot(W) + b) elementwise
            //http://deeplearning.net/software/theano/tutorial/loop.html

            // defining the tensor variables
            var X = T.Matrix<float>("x");
            var W = T.Matrix<float>("W");
            var b_sym = T.Vector<float>("b_sym");

            var results = T.Scan(v => T.Tanh(T.Dot(v, W) + b_sym), sequence: X);
            var compute_elementwise = T.Function(inputs: new[] { X, W, b_sym }, output: results);

            // test values
            var x = NN.Eye<float>(2);
            var w = NN.Ones<float>(2, 2);
            var b = NN.Ones<float>(2);
            b.Item[1] = 2;

            var result = compute_elementwise(new[] { x, w, b });
            var expected = NN.Tanh(x.Dot(w) + b);

            AssertArray.AreAlmostEqual(expected[0], result[0]);
        }

        //[TestMethod]
        //public void TestJacobian()
        //{
        //    var x = T.Tensor("x");
        //    var y = T.Pow(x, 2);
        //    var J = T.Scan((i, y, x) => T.Grad(y[i], x), sequences: Tensor.Range(y.Shape[0]));
        //    var f = T.Function(x, J);

        //    var expected = NN.Array(new float[,]
        //    {
        //        {8, 0 },
        //        {0, 8 }
        //    });
        //    AssertArray.AreEqual(expected, f(Tensor.Vector(4, 4)));
        //}

        [TestMethod]
        public void TestRecursive()
        {
            // http://deeplearning.net/software/theano/tutorial/loop.html
            // define tensor variables
            var X = T.Vector<float>("X");
            var W = T.Matrix<float>("W");
            var b_sym = T.Matrix<float>("b_sym");
            var U = T.Matrix<float>("U");
            var Y = T.Matrix<float>("Y");
            var V = T.Matrix<float>("V");
            var P = T.Matrix<float>("P");
            var results = T.Scan((yy, pp, xx_tm1) => T.Tanh(T.Dot(xx_tm1, W) + T.Dot(yy, U) + T.Dot(pp, V)),
                sequences: new[] { Y, P[XSlicer.Step(-1)] },
                outputsInfo: X);
            var compute_seq = T.Function(inputs: new[] { X, W, Y, U, P, V }, output: results);
            // test values
            var x = NN.Zeros<float>(2);
            x.Item[1] = 1;
            var w = NN.Ones<float>(2, 2);
            var y = NN.Ones<float>(5, 2);
            y.Item[0] = -3;
            var u = NN.Ones<float>(2, 2);
            var p = NN.Ones<float>(5, 2);
            p.Item[0] = 3;
            var v = NN.Ones<float>(2, 2);
            var result = compute_seq(new[] { x, w, y, u, p, v }); // Array<float>[5] => theano returns Array<float>[5][1]
            // comparison with numpy
            var x_res = NN.Zeros<float>(5, 2);
            x_res[0] = NN.Tanh(x.Dot(w) + y[0].Dot(u) + p[4].Dot(v));
            for (int i = 1; i < 5; i++)
                x_res[i] = NN.Tanh(x_res[i - 1].Dot(w) + y[i].Dot(u) + p[4 - i].Dot(v));

            AssertArray.AreAlmostEqual(x_res, result);
        }

        [TestMethod, TestCategory("Uses structural equality")]
        public void SumHasCorrectGrad()
        {
            // sequence of input
            var xs = T.Matrix<float>("xs");
            // accumulator
            var z = T.Vector<float>("z");

            // sum xs in the accumulator
            var partialSums = T.Scan((x, a) => x + a, xs, z);
            // get the last value
            var sum = partialSums[-1];
            var cost = T.Sum(sum * sum);
            var dz = T.Grad(cost, z);

            var slicing = dz as Slicing<float>;
            Assert.AreEqual(1, slicing.Slices.Count);
            Assert.IsTrue(slicing.Slices[0].IsSingleton);
            Assert.AreEqual(-1, ((Scalar<int>.Const)slicing.Slices[0].Start).Value);

            var dfor = slicing.x as Tensor<float>.For;
            var backLoop = dfor.Loop;

            Assert.AreEqual(3, backLoop.Sequences.Count);
            Assert.AreEqual(3, backLoop.Fors.Count);
            Assert.AreEqual(1, dfor.Index);

            var variables = backLoop.Variables.Cast<Tensor<float>>().ToList();
            var x_ = variables[0];
            Assert.AreEqual("x_", x_.Name);
            var a_ = variables[1];
            Assert.AreEqual("a_", a_.Name);
            var delta_a_ = variables[2];
            Assert.AreEqual("delta_a_", delta_a_.Name);
            var dx_ = variables[3];
            Assert.AreEqual("dx_", dx_.Name);
            var da_ = variables[4];
            Assert.AreEqual("da_", da_.Name);
            var dx = (Tensor<float>)backLoop.Fors[0].Expression;
            var da = (Tensor<float>)backLoop.Fors[1].Expression;
            Assert.IsTrue((delta_a_ + da_).StructuralEquality(dx));
            Assert.IsTrue((delta_a_ + da_).StructuralEquality(da));
        }

        [TestMethod, TestCategory("Uses structural equality")]
        public void SumProductHasCorrectGrad()
        {
            // sequence of input
            var xs = T.Matrix<float>("xs");
            // accumulator
            var z = T.Vector<float>("z");

            // sum xs in the accumulator
            Func<Tensor<float>, Tensor<float>, IList<Tensor<float>>> rec = (x, a) =>
                new List<Tensor<float>>() { x + a, x * a };
            var loop = T.Scan(rec, xs, new[] { z, null });

            // get the last value
            var prod = loop[1][-1];
            var cost = T.Sum(prod);
            var dz = T.Grad(cost, z);

            var slicing = dz as Slicing<float>;
            Assert.AreEqual(1, slicing.Slices.Count);
            Assert.IsTrue(slicing.Slices[0].IsSingleton);
            Assert.AreEqual(-1, ((Scalar<int>.Const)slicing.Slices[0].Start).Value);

            var dfor = slicing.x as Tensor<float>.For;
            var backLoop = dfor.Loop;

            Assert.AreEqual(3, backLoop.Sequences.Count);
            Assert.AreEqual(3, backLoop.Fors.Count);
            Assert.AreEqual(1, dfor.Index);

            var variables = backLoop.Variables.Cast<Tensor<float>>().ToList();
            var x_ = variables[0];
            Assert.AreEqual("x_", x_.Name);
            var a_ = variables[1];
            Assert.AreEqual("a_", a_.Name);
            var d_f1_ = variables[2];
            Assert.AreEqual("delta_f1_", d_f1_.Name);
            var da_ = variables[4];
            Assert.AreEqual("da_", da_.Name);
            var dx = (Tensor<float>)backLoop.Fors[0].Expression;
            var da = (Tensor<float>)backLoop.Fors[1].Expression;
            Assert.IsTrue((d_f1_ * a_ + da_).StructuralEquality(dx));
            Assert.IsTrue((d_f1_ * x_ + da_).StructuralEquality(da));
        }

        [TestMethod, TestCategory("Uses structural equality")]
        public void SumProductWithSharedHasCorrectGrad()
        {
            // sequence of input
            var xs = T.Matrix<float>("xs");
            // accumulator
            var z = T.Vector<float>("z");
            var b = T.Vector<float>("b");

            // sum xs in the accumulator
            Func<Tensor<float>, Tensor<float>, IList<Tensor<float>>> rec = (x, a) =>
                new List<Tensor<float>>() { x + a, x * a + b };
            var loop = T.Scan(rec, xs, new[] { z, null });

            // get the last value
            var prod = loop[1][-1];
            var cost = T.Sum(prod);
            //var dz = T.Grad(cost, z);
            var db = T.Grad(cost, b);

            var reshape = db as Reshaping<float>;
            var sum = reshape.x as Sum<float>;
            Assert.AreEqual(0, sum.Axis);

            var dfor = sum.x as Tensor<float>.For;
            var backLoop = dfor.Loop;

            Assert.AreEqual(3, backLoop.Sequences.Count);
            Assert.AreEqual(4, backLoop.Fors.Count);
            Assert.AreEqual(2, dfor.Index);

            // TODO: check why a recursive was expected
            //var db_ = dfor.RecursiveVariable;
            //Assert.AreEqual("db_", db_.Name);

            var variables = backLoop.Variables.Cast<Tensor<float>>().ToList();
            var x_ = variables[0];
            Assert.AreEqual("x_", x_.Name);
            var a_ = variables[1];
            Assert.AreEqual("a_", a_.Name);
            var d_f1_ = variables[2];
            Assert.AreEqual("delta_f1_", d_f1_.Name);
            var da_ = variables[4];
            Assert.AreEqual("da_", da_.Name);
            var dx = (Tensor<float>)backLoop.Fors[0].Expression;
            var da = (Tensor<float>)backLoop.Fors[1].Expression;
            Assert.IsTrue((d_f1_ * a_ + da_).StructuralEquality(dx));
            Assert.IsTrue((d_f1_ * x_ + da_).StructuralEquality(da));
            Assert.IsTrue((d_f1_).StructuralEquality(dfor.Expression));
        }

        [TestMethod, TestCategory("Uses structural equality")]
        public void SumProductWithSharedCanTrain()
        {
            var n = 2;
            // sequence of input
            var xs = T.Matrix<float>("xs");
            // accumulator
            var z = T.Vector<float>("z");
            var b = T.Shared(NN.Ones(n), "b");

            // sum xs in the accumulator
            Func<Tensor<float>, Tensor<float>, IList<Tensor<float>>> rec = (x, a) =>
                new List<Tensor<float>>() { x + a, x * a + b };
            var loop = T.Scan(rec, xs, new[] { z, null });

            // get the last value
            var prod = loop[1][-1];

            // compute the cost and the gradient for the shared b.
            var cost = T.Sum(prod);
            var db = T.Grad(cost, b);

            var costFunction = T.Function(xs, z, output: cost);
            var xs_ = NN.Array(new float[,] {
                { 1, -1 },
                { 0, -2 }
            });

            var z_ = NN.Zeros(n);

            var cost_xs_z = costFunction(xs_, z_);
            Assert.AreEqual(4, cost_xs_z);

            var updates = new OrderedDictionary { { b, b - 0.05f * db } };
            var train = T.Function(xs, z, output: cost, updates: updates);
            var cost_xs_z2 = train(xs_, z_);
            AssertArray.AreAlmostEqual(NN.Array(new[] { 0.95f, 0.95f }), b.Value);
        }

        // TODO: non-sequences + IntArray
        //[TestMethod]
        //public void TestPolynomial()
        //{
        //    var coefficients = T.Vector<float>("coefficients");
        //    var x = T.Scalar<float>("x");
        //    var max_coefficients_supported = 10000;

        //    // Generate the components of the polynomial
        //    var full_range = T.Range(max_coefficients_supported);
        //    var components = T.Scan(fn: (coeff, power, free_var) => coeff * (free_var * *power),
        //                         sequences: new[] { coefficients, full_range },
        //                         non_sequences: x);
        //    var polynomial = components.Sum();
        //    var calculate_polynomial = T.Function(input1: coefficients, input2: x,
        //                               outputs: polynomial);

        //    var test_coeff = NN.Array(new[] { 1f, 0f, 2f });
        //    var result = calculate_polynomial(test_coeff, 3);
        //    AssertArray.AreEqual(19f, result);
        //}

        [TestMethod, TestCategory("Uses structural equality")]
        public void RnnXorHasCorrectGradient()
        {
            NN.Random.Seed(12345);
            int nh = 10; // hidden layer

            var Wbit = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, 1).As<float>(), "Wbit");
            var Wstate = T.Shared(NN.Eye<float>(nh), "Wstate");
            var Wout = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 1, nh).As<float>(), "Wout");
            var b = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, 1).As<float>(), "b");

            var state0 = T.Shared(NN.Zeros<float>(nh, 1), "state0");

            var bits = T.Tensor3<float>("bits");               // n x 1
            var expected = T.Matrix<float>("expected");        // 1 x 1

            Func<Tensor<float>, Tensor<float>, Tensor<float>> recurrence = (bit, oldState) =>
            {
                return T.Tanh(T.Dot(Wbit, bit) + T.Dot(Wstate, oldState) + b);
            };

            var states = T.Scan(fn: recurrence, sequence: bits, outputsInfo: state0);

            var output = T.Tanh(T.Dot(Wout, states[(Slice)(-1)]));
            var error = 0.5f * T.Norm2(output - expected);
            var classify = T.Function(bits, output);
            var gradients = T.Grad(error);

            var gradWstate = gradients[Wstate];
            Assert.IsNotNull(gradWstate);
            var gradWstateIsReshape = gradWstate as Reshaping<float>;
            Assert.IsNotNull(gradWstateIsReshape);
            var gradWstateIsSum = gradWstateIsReshape.x as Sum<float>;
            Assert.IsNotNull(gradWstateIsSum);
            var dfor = gradWstateIsSum.x as Tensor<float>.For;
            var backLoop = dfor.Loop;

            Assert.AreEqual(3, backLoop.Sequences.Count); // bit, states, delta
            Assert.AreEqual(6, backLoop.Fors.Count); // dbit, dstate, dWstate, db, dWbit, dstate_p1
            Assert.AreEqual(3, dfor.Index);

            // TODO: check why a recursive was expected
            //var dWstate_ = dfor.RecursiveVariable;
            //Assert.AreEqual("dWstate_", dWstate_.Name);

            var variables = backLoop.Variables.Cast<Tensor<float>>().ToList();
            var bit_ = variables[0];
            Assert.AreEqual("bit_", bit_.Name);
            var oldState_ = variables[1];
            Assert.AreEqual("oldState_", oldState_.Name);
            var delta_oldState_ = variables[2];
            Assert.AreEqual("delta_oldState_", delta_oldState_.Name);
            var dbit_ = variables[3];
            Assert.AreEqual("dbit_", dbit_.Name);
            var doldState_ = variables[4];
            Assert.AreEqual("doldState_", doldState_.Name);
            var oldState_tp1_ = variables[5];
            Assert.AreEqual("oldState_tp1", oldState_tp1_.Name);

            var d = T.Sum((delta_oldState_ + doldState_) * (1f - T.Square(oldState_tp1_)), axis: 1, keepDims: true);

            var doldState = (Tensor<float>)backLoop.Fors[1].Expression;
            (T.Dot(Wstate, d, transposeX: true)).AssertEqual(doldState);

            var dWstate = (Tensor<float>)backLoop.Fors[3].Expression;
            var dWstateExp = T.Dot(d, oldState_, transposeY: true);
            dWstateExp.AssertEqual(dWstate);

            var dbit = (Tensor<float>)backLoop.Fors[0].Expression;
            (T.Dot(Wbit, d, transposeX: true)).StructuralEquality(dbit);

            var oldState_tp1 = (Tensor<float>)backLoop.Fors[5].Expression;
            oldState_tp1.AssertEqual(oldState_);
        }
    }
}
