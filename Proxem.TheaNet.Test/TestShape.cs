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
using Dim = Proxem.TheaNet.Scalar<int>;
using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestShape
    {
        [TestInitialize]
        public void Initialize()
        {
            Runtime.Reset();
        }

        [TestMethod]
        public void TestShapeOfShared()
        {
            var v = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 13).As<float>(), "v");
            AssertArray.WriteTheSame(new[] { 13 }, v.Shape);

            var M = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 5, 4).As<float>(), "M");
            AssertArray.WriteTheSame(new[] { 5, 4 }, M.Shape);

            var V = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 5, 4, 8).As<float>(), "V");
            AssertArray.WriteTheSame(new[] { 5, 4, 8 }, V.Shape);
        }

        [TestMethod]
        public void TestShapeOfSlice()
        {
            var v = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 13).As<float>(), "v");
            AssertArray.WriteTheSame(new[] { 3 }, v[XSlicer.Range(5, 8)].Shape);
            AssertArray.WriteTheSame(new[] { 3 }, v[XSlicer.Range(8, 5, -1)].Shape);
            AssertArray.WriteTheSame(new[] { 4 }, v[XSlicer.Range(3, 11, 2)].Shape);

            var M = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 8, 22).As<float>(), "M");
            AssertArray.WriteTheSame(new[] { 5, 17 }, M[XSlicer.Range(6, 1, -1), XSlicer.From(5)].Shape);
        }

        [TestMethod]
        public void TestShapeOfDot()
        {
            var v1 = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 13).As<float>(), "v1");
            var v2 = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 13).As<float>(), "v2");
            AssertArray.WriteTheSame(new[] { 13, 13 }, T.Dot(v1, v2, transposeY: true).Shape);
            AssertArray.AreEqual(EmptyArray<Dim>.Value, T.Dot(v1, v2).Shape);
            AssertArray.AreEqual(EmptyArray<Dim>.Value, T.Dot(v1, v2, transposeX: true).Shape);

            var M1 = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 5, 13).As<float>(), "M1");
            AssertArray.WriteTheSame(new[] { 5 }, T.Dot(M1, v1).Shape);

            var M2 = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 13, 7).As<float>(), "M2");
            AssertArray.WriteTheSame(new[] { 5, 7 }, T.Dot(M1, M2).Shape);

            var M3 = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 8, 13).As<float>(), "M3");
            AssertArray.WriteTheSame(new[] { 5, 8 }, T.Dot(M1, M3, transposeY: true).Shape);
        }

        [TestMethod, TestCategory("Exception")]
        public void WrongDotThrowsRankException()
        {
            var v1 = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 13).As<float>(), "v1");
            var v2 = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 8).As<float>(), "v2");
            AssertThrows<RankException>(() => T.Dot(v1, v2, transposeX: true, transposeY: true));
            AssertThrows<RankException>(() => T.Dot(v1, v2));

            var M1 = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 5, 15).As<float>(), "M");
            AssertThrows<RankException>(() => T.Dot(M1, v1));

            var M2 = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 13, 7).As<float>(), "M");
            AssertThrows<RankException>(() => T.Dot(M1, M2));
        }

        [TestMethod]
        public void TestShapeOfScanWithShared()
        {
            var W = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 5, 4).As<float>(), "W");

            Func<Tensor<float>, Tensor<float>, Tensor<float>> recurrence = (x, acc) =>
            {
                return acc + T.Dot(W, x);
            };

            var X = T.Shared(NN.Zeros(13, 4), "X");
            var acc0 = T.Shared(NN.Zeros<float>(5), "acc0");

            var loop = T.Scan(fn: recurrence, sequences: new[] { X }, outputsInfo: acc0);
            AssertArray.WriteTheSame(new [] {13, 5}, loop.Shape);

            var norm2 = T.Norm2(loop[-1]);
            var grad = T.Grad(norm2, W);
            AssertArray.WriteTheSame(new[] { 5, 4 }, grad.Shape);
        }

        [TestMethod]
        public void TestShapeOfScan()
        {
            var W = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 5, 4).As<float>(), "W");

            var X = T.Matrix<float>("X");
            var acc0 = T.Shared(NN.Zeros<float>(5), "acc0");

            var loop = T.Scan(fn: (x, acc) => acc + T.Dot(W, x), sequence: X, outputsInfo: acc0);
            AssertArray.WriteTheSame(new[] { X.Shape[0], 5 }, loop.Shape);

            var norm2 = T.Norm2(loop[-1]);
            var grad = T.Grad(norm2, W);
            grad.Shape.WillEqualTo(new Dim[] { 5, 4 });
        }

        [TestMethod]
        public void TestDropAt()
        {
            Dim d0 = 10, d1 = 11, d2 = 12, d3 = 13;
            var shape = new [] { d0, d1, d2, d3 };

            AssertArray.AreEqual(new[] { d1, d2, d3 }, shape.DropAt(0));
            AssertArray.AreEqual(new[] { d0, d2, d3 }, shape.DropAt(1));
            AssertArray.AreEqual(new[] { d0, d3 }, shape.DropAt(1, 2));
        }

        public void AssertThrows<T>(Action f) where T: Exception
        {
            try
            {
                f();
                throw new Exception($"No {typeof(T).ToString()} was thrown");
            }
            catch(T)
            {
            }
        }
    }
}
