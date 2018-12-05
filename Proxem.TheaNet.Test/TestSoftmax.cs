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
using static Proxem.NumNet.Slicer;

using real = System.Single;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestSoftmax
    {
        public Tensor<real> CategoricalCrossentropy(Tensor<real> coding_dist, Tensor<real> true_dist)
        {
            return -T.Sum(true_dist * T.Log(coding_dist), axis: 1, keepDims: true);
        }

        [TestMethod]
        public void TestMethod1()
        {
            // https://github.com/Theano/Theano/issues/3162
            // When using unbounded activation functions (e.g. Relu) the softmax function can saturate. This can lead to nan gradients when paired with categorical crossentropy cost.
            // If the softmax function is replaced with a numerically stable version of log-softmax and this is used directly in the cost function, then the gradients don't blow up.
            // It seems that this could be implemented as a pattern to recognize(softmax paired with categorical crossentropy).
            // Here's a code snippet that illustrates the problem with the regular softmax versus doing the same thing with the numerically stable log-softmax, where the former gives nans in the gradient and the latter does not blow up. It's interesting because the experiment indicates that for the regular softmax case, the crossentropy loss is coming out numerically stable but not the gradient.

            Binding.Compiler.Debug = true;

            var x = T.Matrix<real>("x");
            var y = T.Matrix<real>("y");

            // regular softmax and crossentropy
            var sm = T.Softmax(x);
            var cm1 = CategoricalCrossentropy(sm, y);
            var g1 = T.Grad(T.Mean(cm1), x);

            // numerically stable log-softmax with crossentropy
            var xdev = x - T.Max(x, axis: 1, keepDims: true);
            var lsm = xdev - T.Log(T.Sum(T.Exp(xdev), axis: 1, keepDims: true));
            //var lsm2 = xdev - T.LogSumExp(xdev, axis: 1, keepDims: true);
            var sm2 = T.Exp(lsm); // just used to show equivalence with sm
            var cm2 = -T.Sum(y * lsm, axis: 1);
            var g2 = T.Grad(T.Mean(cm2), x);

            // create some inputs into a softmax that are large and labels
            var large = 1f; // 10f
            var a = NN.Exp(NN.Random.Uniform<float>(0, large,  5, 10));
            // create some one-hot coded labels
            var b = NN.Zeros<float>(5, 10);
            b[Range(0, 5), Range(0, 5)] = NN.Eye<float>(5);

            // show equivalence of softmax and exponentiated numerically stable log-softmax
            var f1 = T.Function(input: x, output1: sm, output2: sm2);
            var sm_ = f1(a);
            var sm_1 = sm_.Item1;       // classical softmax
            var sm_2 = sm_.Item2;       // log(sum(exp)) softmax
            AssertArray.AreAlmostEqual(sm_1, sm_2);

            // now show that the two versions result in the same crossentropy cost
            // this indicates that the forward function does provide some numerical stability
            var f2 = T.Function(input1: x, input2: y, output1: cm1, output2: cm2);
            var c_ = f2(a, b);
            var c_1 = c_.Item1;
            var c_2 = c_.Item2;
            AssertArray.AreAlmostEqual(c_1, c_2);

            // now, show that in the standard softmax case the gradients blow up 
            // while in the log-softmax case they don't
            var f3 = T.Function(input1: x, input2: y, output1: g1, output2: g2);
            var g_ = f3(a, b);
            var g_1 = g_.Item1;
            var g_2 = g_.Item2;
            Assert.IsTrue(float.IsNaN(g_1.Sum()));
            Assert.IsFalse(float.IsNaN(g_2.Sum()));
        }
    }
}
