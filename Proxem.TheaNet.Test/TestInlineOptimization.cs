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
using Proxem.TheaNet.Operators.Tensors;

using static Proxem.NumNet.Slicer;

namespace Proxem.TheaNet.Test
{
    using Dim = Scalar<int>;

    [TestClass]
    public class TestInlineOptimization
    {
        [TestMethod]
        public void SimplifiesAdditionOfScalars()
        {
            var x = Op.Scalar<int>("x");
            Assert.AreEqual(x, x + 0);
            Assert.AreEqual(x, (x + 1) - 1);
            Assert.AreEqual(x, (1 + x) - 1);
            Assert.AreEqual(x, (x - 1) + 1);
        }

        [TestMethod]
        public void SimplifiesMultiplicationOfScalars()
        {
            var x = Op.Scalar<float>("x");
            Assert.AreEqual(x, x * 1);
            Assert.AreEqual(x, (x * 2) * 0.5f);
            Assert.AreEqual(x, (x * 2) / 2);
            Assert.AreEqual(x, (2 * x) * 0.5f);
            Assert.AreEqual(x, (2 * x) / 2);
            Assert.AreEqual(x, (x * 0.5f) * 2);
            Assert.AreEqual(x, (x / 2) * 2);
        }

        [TestMethod]
        public void SimplifiesMultiplicationOfOneHot()
        {
            var x = Op.Scalar<float>("x");
            var oneHotPoint = Op.OneHot<float>(new Dim[] { 10 }, new Dim[] { 2 }, 1f);
            Assert.IsTrue(x * oneHotPoint is OneHotPoint<float>);
            Assert.AreEqual(x, (x * oneHotPoint as OneHotPoint<float>).Content);
        }

        [TestMethod]
        public void SimplifiesIndexingOfConst()
        {
            var x = Op.Const(5, 10, 20);
            Assert.IsTrue(x[0] is Fill<int>);
            Assert.IsTrue(x[5] is Fill<int>);
            Assert.IsTrue(x[5, 2] is Fill<int>);

            Assert.IsTrue(x[From(1)] is Fill<int>);
            Assert.IsTrue(x[5, From(1)] is Fill<int>);

            Assert.IsTrue(x[Op.Range(5)] is Fill<int>);

            Assert.IsTrue(x.Item[5, 2] is Scalar<int>.Const);
        }
    }
}
