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
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestEinsteinSum
    {
        [TestInitialize]
        public void Reset()
        {
            Runtime.Reset();
        }

        [TestMethod]
        public void TestEinstein_bi_io_bo()
        {
            var x = T.Matrix<float>(20, 10, "x");
            var y = T.Matrix<float>(10, 50, "y");

            var eins = T.EinsteinSum(x, y, "bi,io->bo");
            // checks that the outputed shape is correct
            eins.AssertOfShape(20, 50);

            // checks that the gradient shapes are correct
            var loss = T.Sum(eins);
            T.Grad(loss);
        }

        [TestMethod]
        public void TestEinstein_bo_bi_io()
        {
            var x = T.Matrix<float>(20, 10, "x");
            var y = T.Matrix<float>(20, 50, "y");

            var eins = T.EinsteinSum(x, y, "bo,bi->io");
            eins.AssertOfShape(50, 10);

            var loss = T.Sum(eins);
            T.Grad(loss);
        }
    }
}
