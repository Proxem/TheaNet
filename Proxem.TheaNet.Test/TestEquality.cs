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

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestEquality
    {
        [TestMethod, TestCategory("Not implemented")]
        public void WorksOnAddFloatArray()
        {
            var x = Op.Vector<float>("x");
            var y = Op.Vector<float>("y");
            var y2 = Op.Vector<float>("y2");

            var z = x + y;
            Assert.IsTrue((x + y).StructuralEquality(z));
            Assert.IsFalse((x - y).StructuralEquality(z));
            Assert.IsFalse((x * y).StructuralEquality(z));
            Assert.IsFalse((x + y2).StructuralEquality(z));
        }

        [TestMethod, TestCategory("Not implemented")]
        public void WorksOnFill()
        {
            var x = Op.Vector<float>("x");
            var xFilled1 = Op.OnesLike(x);
            var xFilled1_bis = Op.OnesLike(x);
            var xFilled0 = Op.ZerosLike(x);

            Assert.IsTrue(xFilled1.StructuralEquality(xFilled1_bis));
            Assert.IsFalse(xFilled1.StructuralEquality(xFilled0));
            Assert.IsFalse(xFilled1.StructuralEquality(x));
        }

        [TestMethod, TestCategory("Not implemented")]
        public void WorksOnAddFloat()
        {
            var x = Op.Scalar<float>("x");
            var y = Op.Scalar<float>("y");
            var y2 = Op.Scalar<float>("y2");

            var z = x + y;
            Assert.IsTrue((x + y).StructuralEquality(z));
            Assert.IsFalse((x - y).StructuralEquality(z));
            Assert.IsFalse((x * y).StructuralEquality(z));
            Assert.IsFalse((x + y2).StructuralEquality(z));
        }

        [TestMethod, TestCategory("Not implemented")]
        public void WorksOnAddConstToFloatArray()
        {
            var x = Op.Vector<float>("x");
            var z = x + 1;
            Assert.IsTrue((x + 1).StructuralEquality(z));
            Assert.IsFalse((x - 1).StructuralEquality(z));
            Assert.IsFalse((x * 1).StructuralEquality(z));
            Assert.IsFalse((x + 1.5f).StructuralEquality(z));
        }
    }
}
