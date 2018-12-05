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
using System.Collections.Specialized;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Proxem.NumNet;
using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestShared
    {
        [TestInitialize]
        public void Initialize()
        {
            Runtime.Reset();
        }

        // https://github.com/goodfeli/theano_exercises/blob/master/01_basics/02_compiling_and_running/02_shared_soln.py
        public Tensor<float>.Shared make_shared(string name, params int[] shape)
        {
            //Returns a theano shared variable containing a tensor of the specified shape.
            //You can use any value you want.
            return T.Shared(NN.Zeros<float>(shape), name);
        }

        public void exchange_shared(Tensor<float>.Shared a, Tensor<float>.Shared b)
        {
            //a: a theano shared variable
            //b: a theano shared variable
            //Uses get_value and set_value to swap the values stored in a and b
            var temp = a.Value;
            a.Value = b.Value;
            b.Value = temp;
        }

        public Action make_exchange_func(Tensor<float>.Shared a, Tensor<float>.Shared b)
        {
            //a: a theano shared variable
            //b: a theano shared variable
            //Returns f
            //where f is a theano function, that, when called, swaps the
            //values in a and b
            //f should not return anything

            var f = T.Function(updates: new OrderedDictionary {
                { a, b },
                { b, a }
            });
            return f;
        }

        [TestMethod]
        public void CanSwapTwoShared()
        {
            var a = make_shared("a", 5, 4, 3);
            Assert.IsTrue(a.Value.Shape.Zip(new int[] { 5, 4, 3 }, (x, y) => x == y).All(x => x));
            var b = make_shared("b", 5, 4, 3);
            Assert.IsTrue(b.Value.Shape.Zip(new int[] { 5, 4, 3 }, (x, y) => x == y).All(x => x));
            a.Value = NN.Zeros<float>(5, 4, 3);
            b.Value = NN.Ones<float>(5, 4, 3);
            exchange_shared(a, b);
            Assert.IsTrue(a.Value.All(x => x == 1.0f));
            Assert.IsTrue(b.Value.All(x => x == 0.0f));
            var f = make_exchange_func(a, b);
            f();
            Assert.IsTrue(a.Value.All(x => x == 0.0f));
            Assert.IsTrue(b.Value.All(x => x == 1.0f));
        }

        [TestMethod]
        public void TestFunctionUsesActualValueOfShared()
        {
            // p. 426 "Using SharedVariables as pfunc Parameters"
            var a = T.Scalar<float>("a");
            var b = T.Shared(7f, "b");

            // create two functions that use ‘b‘ as an implicit input
            var f1 = T.Function(a, a + b);
            var f2 = T.Function(a, a * b);
            Assert.AreEqual(12, f1(5));

            b.Value = 8; // modify the shared variable’s value
            Assert.AreEqual(13, f1(5)); // the new value is reflected in any compiled functions
            Assert.AreEqual(32, f2(4)); // f2 uses the latest value in b’s container
        }

        [TestMethod]
        public void TestInt()
        {
            var state = T.Shared(0, "state");
            var inc = T.Scalar<int>("inc");

            var updates = new OrderedDictionary { { state, state + inc } };
            var accumulator = T.Function(inc, state, updates);

            Assert.AreEqual(0, state.Value);
            Assert.AreEqual(0, accumulator(1));
            Assert.AreEqual(1, state.Value);
            Assert.AreEqual(1, accumulator(300));
            Assert.AreEqual(301, state.Value);

            state.Value = -1;
            Assert.AreEqual(-1, accumulator(3));
            Assert.AreEqual(2, state.Value);

            var updates2 = new OrderedDictionary { { state, state - inc } };
            var decrementor = T.Function(inc, state, updates2);

            Assert.AreEqual(2, decrementor(2));
            Assert.AreEqual(0, state.Value);
        }

        [TestMethod]
        public void TestFloat()
        {
            var state = T.Shared(0f, "state");
            var inc = T.Scalar<float>("inc");

            var updates = new OrderedDictionary { { state, state + inc } };
            var accumulator = T.Function(inc, state, updates);

            Assert.AreEqual(0, state.Value);
            Assert.AreEqual(0, accumulator(1));
            Assert.AreEqual(1, state.Value);
            Assert.AreEqual(1, accumulator(300));
            Assert.AreEqual(301, state.Value);

            state.Value = -1;
            Assert.AreEqual(-1, accumulator(3));
            Assert.AreEqual(2, state.Value);

            var updates2 = new OrderedDictionary { { state, state - inc } };
            var decrementor = T.Function(inc, state, updates2);

            Assert.AreEqual(2, decrementor(2));
            Assert.AreEqual(0, state.Value);
        }
    }
}
