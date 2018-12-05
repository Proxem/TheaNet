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
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Proxem.NumNet;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestScanGrad
    {
        [TestInitialize]
        public void Initialize()
        {
            Runtime.Reset();
        }

        [TestMethod]
        public void TestLoopInvariantCodeMotion()
        {
            var y = T.Vector<float>("y");
            var z = T.Vector<float>("z");

            Func<Tensor<float>, Tensor<float>, Tensor<float>> recurrence = (x, acc) =>
            {
                return acc + x + y * z;
            };

            var X = T.Matrix<float>("X");
            var acc0 = T.Shared(NN.Zeros<float>(4), "acc0");

            var result = T.Scan(fn: recurrence, sequences: new[] { X }, outputsInfo: acc0);
            var norm2 = T.Norm2(result[-1]);

            var f = T.Function(new[] { X, y, z }, norm2);
        }

        [TestMethod]
        public void TestScanOnSumDot()
        {
            var W = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 4, 5).As<float>(), "W");
            var X = T.Matrix<float>("X");
            var acc0 = T.Shared(NN.Zeros<float>(4), "acc0");

            var result = T.Scan(fn: (x, acc) => acc + T.Dot(W, x), sequences: new[] { X }, outputsInfo: acc0);
            var norm2 = T.Norm2(result[-1]);

            var f = T.Function(X, norm2);
            var X0 = NN.Random.Uniform(-1f, 1f, 10, 5);
            f(X0);

            var grad = T.Grad(norm2, W);
            var df = T.Function(X, grad);
            df(X0);
            //var df = T.Function(X, new[] { norm2, grad });
        }

        [TestMethod]
        public void TestScanOnTanhSumDot()
        {
            var W = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 4, 5).As<float>(), "W");

            Func<Tensor<float>, Tensor<float>, Tensor<float>> recurrence =
                (x, acc) => T.Tanh(acc + T.Dot(W, x));

            var X = T.Matrix<float>(-1, 5, "X");
            var acc0 = T.Shared(NN.Zeros<float>(4), "acc0");

            var result = T.Scan(fn: recurrence, sequences: new[] { X }, outputsInfo: acc0);
            var norm2 = T.Norm2(result[-1]);

            var f = T.Function(X, norm2);

            var grad = T.Grad(norm2, W);

            var df = T.Function(X, output1: norm2, output2: grad);
            df(NN.Array(new[,] { { 0f, 0f, 0f, 0f, 0f } }));

            AssertTensor.PassesGradientCheck(X, norm2, acc0);
            AssertTensor.PassesGradientCheck(X, norm2, W);
        }

        [TestMethod]
        public void ScanPassesGradientCheckOnSeed()
        {
            var n = 5;
            var epsilon = 0.01f;

            var zero = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, n).As<float>(), "zero");
            var xs = T.Matrix<float>(-1, n, "xs");

            var sum = T.Scan((x, acc) => acc + x, sequence: xs, outputsInfo: zero)[-1];
            var norm2 = T.Norm2(sum);

            var checkManually = T.RandomGradientCheck(xs, norm2, zero, computed: 2 * sum);
            for (int _ = 0; _ < 50; ++_)
            {
                var xs_ = NN.Random.Uniform(-1, 1, 10, n).As<float>();
                var checkRes = checkManually(xs_, epsilon);
                var finite = checkRes.Item1;
                var backpropagated = checkRes.Item2;
                AssertArray.WithMessage("GradientCheck isn't precise enough", () =>
                    AssertArray.AreAlmostEqual(finite, backpropagated, relativeErr: 1e-3f, absErr: 1e-4f)
                );
            };

            var checkGrad = T.RandomGradientCheck(xs, norm2, zero);
            for (int _ = 0; _ < 50; ++_)
            {
                var xs_ = NN.Random.Uniform(-1, 1, 10, n).As<float>();
                var checkRes = checkGrad(xs_, epsilon);
                var finite = checkRes.Item1;
                var backpropagated = checkRes.Item2;
                AssertArray.WithMessage("Backward isn't precise enough", () =>
                    AssertArray.AreAlmostEqual(finite, backpropagated, relativeErr: 1e-3f, absErr: 1e-4f)
                );
            };
        }

        [TestMethod]
        public void ScanPassesGradientCheckOnRec()
        {
            var n = 5;
            var zero = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, n).As<float>(), "zero");
            var b = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, n).As<float>(), "b");
            var xs = T.Matrix<float>(-1, n, "xs");

            var sum = T.Scan((x, acc) => acc + x + b, sequence: xs, outputsInfo: zero)[-1];
            var norm2 = T.Norm2(sum);

            var epsilon = 0.001f;
            var checkManually = T.RandomGradientCheck(xs, norm2, b, computed: 2 * xs.Shape[0].As<float>() * sum);
            for (int _ = 0; _ < 50; ++_)
            {
                var xs_ = NN.Random.Uniform(-1, 1, 10, n).As<float>();
                var checkRes = checkManually(xs_, epsilon);
                var finite = checkRes.Item1;
                var backpropagated = checkRes.Item2;
                AssertArray.WithMessage("GradientCheck isn't precise enough", () =>
                    AssertArray.AreAlmostEqual(finite, backpropagated, relativeErr: 1e-3f, absErr: 1e-4f)
                );
            };

            var checkGrad = T.RandomGradientCheck(xs, norm2, b);
            for (int _ = 0; _ < 50; ++_)
            {
                var xs_ = NN.Random.Uniform(-1, 1, 10, n).As<float>();
                var checkRes = checkGrad(xs_, epsilon);
                var finite = checkRes.Item1;
                var backpropagated = checkRes.Item2;
                AssertArray.WithMessage("Backward isn't precise enough", () =>
                    AssertArray.AreAlmostEqual(finite, backpropagated, relativeErr: 1e-3f, absErr: 1e-4f)
                );
            };
        }

        [TestMethod]
        public void ScanPassesGradientCheckOnSeq2Seq()
        {
            int embeddingSize = 10, vocabSize = 100;

            var L = T.Shared(NN.Random.Uniform(-0.01f, 0.01f, vocabSize, embeddingSize), "L");
            var W = T.Shared(NN.Random.Uniform(-0.01f, 0.01f, embeddingSize, embeddingSize), "W");
            var ids = T.Vector<int>(-1, "ids");
            var xs = L[ids];

            var scan = T.Scan((x, acc) => T.Tanh(T.Dot(acc + x, W)), sequence: xs, outputsInfo: T.Zeros<float>(embeddingSize));
            var norm2 = T.Norm2(scan);

            var grad = T.Grad(norm2);
            var updates = new OrderedDictionary {[W] = W - 0.001f * grad[W], [L] = L - 0.001f * grad[L] };

            var f = T.Function(input: ids, output: norm2, updates: updates);

            Func<Array<int>> init = () => NN.Random.Uniform(0, vocabSize - 1, 10).As<int>();
            f(init());

            AssertTensor.PassesGradientCheck(ids, norm2, W, init: init);
            AssertTensor.PassesGradientCheck(ids, norm2, L, init: init);
        }
    }
}
