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
using Proxem.TheaNet;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestIntExpr
    {
        [TestMethod]
        public void TestAdd()
        {
            var x = T.Scalar<int>("x");
            var y = T.Scalar<int>("y");
            var e = x + y;

            var f = T.Function(input: (x, y), output: e);
            Assert.AreEqual(8, f(5, 3));
        }

        [TestMethod]
        public void TestAdd2()
        {
            var x = T.Scalar<int>("x");
            var e = x + x;

            var f = T.Function(x, e);
            Assert.AreEqual(8, f(4));
        }

        [TestMethod]
        public void TestTwoVarExpr()
        {
            var x = T.Scalar<float>("x");
            var y = T.Scalar<float>("y");
            var e = 2 * x + 3 * y;

            var f = T.Function(input: (x, y), output: e);
            Assert.AreEqual(22, f(5, 4));
        }

        [TestMethod]
        public void TestRange()
        {
            var i = T.Scalar<int>("i");
            var x = T.Range(i);

            var f = T.Function(i, x);

            AssertArray.AreEqual(NN.Range(10), f(10));
        }

        [TestMethod]
        public void MeanAcceptIntTensor()
        {
            var i = T.Scalar<int>("i");
            var x = T.Range(i);
            var y = T.Mean(x);

            var f = T.Function(i, y);

            AssertArray.AreAlmostEqual(f(10), 4.5f);
        }
    }
}
