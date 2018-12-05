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
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Proxem.NumNet;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestGivens
    {
        [TestInitialize]
        public void Initialize()
        {
            Runtime.Reset();
        }

        [TestMethod]
        public void TestGivenIntShared()
        {
            var x = T.Scalar<int>("x");
            var y = T.Shared(3, "y");
            var output = x + y;
            var f = T.Function(input: x, output: output, givens: new OrderedDictionary { { y, 4 } });
            AssertArray.AreEqual(f(2), 6);

            var f2 = T.Function(input: x, output: output, givens: new OrderedDictionary { { y, x + 4 } });
            AssertArray.AreEqual(f2(2), 8);
        }

        [TestMethod]
        public void TestGivenIntVar()
        {
            var x = T.Scalar<int>("x");
            var y = T.Scalar<int>("y");
            var output = x + y;
            var f = T.Function(input: x, output: output, givens: new OrderedDictionary { { y, 4 } });
            AssertArray.AreEqual(f(2), 6);

            var f2 = T.Function(input: x, output: output, givens: new OrderedDictionary { { y, x + 4 } });
            AssertArray.AreEqual(f2(2), 8);
        }

        [TestMethod]
        public void TestGivenFloatShared()
        {
            var x = T.Scalar<float>("x");
            var y = T.Shared(3f, "y");
            var output = x + y;
            var f = T.Function(input: x, output: output, givens: new OrderedDictionary { { y, 4f } });
            AssertArray.AreAlmostEqual(f(2), 6f);

            var f2 = T.Function(input: x, output: output, givens: new OrderedDictionary { { y, x + 4f } });
            AssertArray.AreAlmostEqual(f2(2), 8f);
        }

        [TestMethod]
        public void TestGivenFloatVar()
        {
            var x = T.Scalar<float>("x");
            var y = T.Scalar<float>("y");
            var output = x + y;
            var f = T.Function(input: x, output: output, givens: new OrderedDictionary { { y, 4f } });
            AssertArray.AreAlmostEqual(f(2), 6f);

            var f2 = T.Function(input: x, output: output, givens: new OrderedDictionary { { y, x + 4f } });
            AssertArray.AreAlmostEqual(f2(2), 8f);
        }


        [TestMethod, ExpectedException(typeof(InvalidCastException)), TestCategory("Exception")]
        public void FailGivenCast()
        {
            var x = T.Scalar<int>("x");
            var y = T.Shared(3, "y");
            var output = x + y;
            var f = T.Function(input: x, output: output, givens: new OrderedDictionary { { y, 4.5f } });
            AssertArray.AreEqual(f(2), 6);
        }
    }
}
