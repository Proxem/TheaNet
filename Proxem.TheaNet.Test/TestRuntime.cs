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
using Microsoft.VisualStudio.TestTools.UnitTesting;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestRuntime
    {
        [TestMethod]
        public void CustomOpSupportsStatic()
        {
            var x = T.Scalar<float>("x");
            Scalar<float> y = CustomOp.Create("myCustomCosinus", Cos, x);
            var f = T.Function(x, y);

            AssertAreCoherents(Cos, f);
        }

        private static float Cos(float a) => (float)Math.Cos(a);

        [TestMethod]
        public void CustomOpSupportsLambda()
        {
            var x = T.Scalar<float>("x");
            Scalar<float> y = CustomOp.Create("myCustomSinus", a => (float)Math.Sin(a), x);
            var f = T.Function(x, y);

            AssertAreCoherents(a => (float)Math.Sin(a), f);
        }

        [TestMethod]
        public void CustomOpCanBeDerivated()
        {
            var x = T.Scalar<float>("x");

            Scalar<float> y = CustomOp.Create("myCustomSinus",
                f: a => (float)Math.Sin(a),
                df_dx: (a, b) => CustomOp.Create("myCustomCosinus", Cos, a),
                x: x
            );

            // f = d/dx(sin(x)) = cos(x)
            var f = T.Function(x, T.Grad(y, x));

            AssertAreCoherents(Cos, f);
        }

        private static void AssertAreCoherents(Func<float, float> f1, Func<float, float> f2)
        {
            Assert.AreEqual(f1(1f), f2(1f));
            Assert.AreEqual(f1(3.14f), f2(3.14f));
            Assert.AreEqual(f1(-3.14f / 4), f2(-3.14f / 4));
        }
    }
}
