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
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Proxem.TheaNet.Samples;

namespace Proxem.TheaNet.Test
{
    using NumNet;
    using Binding;
    using T = Op;

    [TestClass]
    public class TestPatch
    {
        [TestMethod]
        public void PatchTraversesNN()
        {
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, 10, 10), "W");
            var x = T.Vector<float>(10, "x");

            var loss = T.Norm2(T.Tanh(T.Dot(W, T.Tanh(T.Dot(W, x)))));

            var lossVar = loss.FindAll<Tensor<float>.Symbol>();
            Assert.AreEqual(2, lossVar.Count);
            Assert.IsTrue(lossVar.Contains(W));
            Assert.IsTrue(lossVar.Contains(x));

            var y = T.Vector<float>("y");
            var V = T.Matrix<float>("V");

            var patchedLoss = (Scalar<float>) loss.Patch(new Patch {[x] = y,[W] = V });
            var patchedLossVar = patchedLoss.FindAll<Tensor<float>.Symbol>();
            Assert.AreEqual(2, patchedLossVar.Count);
            Assert.IsTrue(patchedLossVar.Contains(V));
            Assert.IsTrue(patchedLossVar.Contains(y));

            loss = (Scalar<float>)patchedLoss.Patch(new Patch {[y] = x,[V] = W });
            lossVar = lossVar = loss.FindAll<Tensor<float>.Symbol>();
            Assert.AreEqual(2, lossVar.Count);
            Assert.IsTrue(lossVar.Contains(W));
            Assert.IsTrue(lossVar.Contains(x));
        }

        [TestMethod]
        public void PatchTraversesScan()
        {
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, 10, 10), "W");
            var a = T.Shared(NN.Random.Uniform(-1f, 1f, 10), "a");
            var x = T.Matrix<float>(-1, 10, "x");

            var loss = T.Norm2(T.Scan(x_ => T.Tanh(T.Dot(W, x_) + a), x));

            var lossVar = loss.FindAll<Tensor<float>.Symbol>();
            Assert.AreEqual(4, lossVar.Count);
            Assert.IsTrue(lossVar.Contains(W));
            Assert.IsTrue(lossVar.Contains(x));
            Assert.IsTrue(lossVar.Contains(a));
            var _x_ = lossVar.First(v => v.Name == "x_");

            var y = T.Vector<float>("y");
            var V = T.Matrix<float>("V");
            var b = T.Vector<float>("b");

            var patchedLoss = loss.Patch(new Patch {[x] = y, [W] = V, [a]=b });
            var patchedLossVar = patchedLoss.FindAll<Tensor<float>.Symbol>();
            Assert.AreEqual(4, patchedLossVar.Count);
            Assert.IsTrue(patchedLossVar.Contains(V));
            Assert.IsTrue(patchedLossVar.Contains(y));
            Assert.IsTrue(patchedLossVar.Contains(b));
            var _x_2 = patchedLossVar.First(v => v.Name.StartsWith("x_"));
            Assert.AreNotEqual(_x_, _x_2);
        }

        [TestMethod]
        public void PatchDoesntCreateNewLoopIfNotNeeded()
        {
            var W = T.Shared(NN.Random.Uniform(-1f, 1f, 10, 10), "W1");
            var a = T.Shared(NN.Random.Uniform(-1f, 1f, 10), "a1");
            var b = T.Shared(NN.Random.Uniform(-1f, 1f, 10), "b1");
            var x = T.Matrix<float>(-1, 10, "x");

            var @for = T.Scan(x_ => T.Tanh(T.Dot(W, x_) + a), x);
            var loss = T.Norm2(@for[-1] + b);

            var patchedLoss = loss.Patch(new Patch {[b] = a });
            var patchedFor = patchedLoss.FindAll<Tensor<float>.For>();

            Assert.AreEqual(1, patchedFor.Count);
            Assert.AreEqual(@for, patchedFor.Single());
        }

        [TestMethod]
        public void ConstsArePatched()
        {
            var c = T.Const(1f);
            var x = T.Scalar<float>("x");

            var patchedC = (Scalar<float>)c.Patch(new Patch { [c] = x });
            Assert.AreEqual(x, patchedC);
        }
    }
}
