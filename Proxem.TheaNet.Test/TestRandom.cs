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
using Proxem.NumNet;

using T = Proxem.TheaNet.Op;
using static Proxem.TheaNet.Operators.RandomFactory;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestRandom
    {
        [TestMethod]
        public void TestRandomUniform()
        {
            var seed = 1234;
            var eps = 1;

            NN.Random.Seed(seed);
            var u_exp = NN.Random.Uniform(-eps, eps, 10, 10).As<float>();

            var eps_ = T.Scalar<float>("eps");
            var u = T.Random.Uniform(-eps_, eps_, 10, 10);
            var fu = T.Function(input: eps_, output: u);

            NN.Random.Seed(seed);
            var u_actual = fu(eps);

            AssertArray.AreEqual(u_exp, u_actual);
        }
    }
}
