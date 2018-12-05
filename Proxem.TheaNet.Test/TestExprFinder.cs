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
using Proxem.TheaNet.Binding;
using Proxem.TheaNet.Samples;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestExprFinder
    {
        [TestInitialize]
        public void Init()
        {
            Runtime.Reset();
        }

        [TestMethod]
        public void FindsAllSharedOfElman()
        {
            var elman = new Elman(10, 10, 5, 5, 2);

            var shared = elman.Loss.FindAll<Tensor<float>.Shared>();
            var @params = elman.@params;

            Assert.AreEqual(@params.Length, shared.Count);
            foreach (var w in @params)
                Assert.IsTrue(shared.Contains(w));
        }
    }
}
