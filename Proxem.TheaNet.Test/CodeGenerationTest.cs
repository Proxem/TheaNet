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

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Proxem.NumNet;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Proxem.TheaNet.Binding;

namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class CodeGenerationTest
    {
        [TestInitialize]
        public void SetCompilerFactory()
        {
            FunctionBinder.CompilerFactory = () => new Compiler { Verbose = false };
        }

        [TestMethod]
        public void DoesntInlineAggregator()
        {
            var grad = Op.Shared<float>(NN.Random.Uniform(-1, 1, 10, 20).As<float>(), "Grad");
            var hist = Op.Shared<float>(NN.Random.Uniform(-1, 1, 10, 20).As<float>(), "Hist");
            var meanGrad = grad / Op.Norm2(grad);
            var u = new OrderedDictionary();
            u[hist] = hist + meanGrad * meanGrad;

            var update = Op.Function(u);
            // update();

            // check that the norm2 of grad isn't captured inside the apply
            AssertSourceContains("Norm2", 1);
            var source = FunctionBinder.Compiler.GetSource();
            var norm2 = source.IndexOf("Norm2");
            var apply = source.IndexOf("Apply");
            if (norm2 > apply)
                throw new Exception("Norm2 should be computed before calling Apply");
        }

        [TestMethod]
        public void ComputesSigmoidOnce()
        {
            var x = Op.Vector<float>("x");
            var y = Op.Sigmoid(x);
            var cost = 0.5f * Op.Norm2(y);
            var g = Op.Grad(cost, x);

            var f = Op.Function(input: x, output: (cost, g));
            AssertSourceContains("Sigmoid", exactly: 1);
        }

        [TestMethod]
        public void DontDuplicatesGradient()
        {
            var x = Op.Vector<float>("x");
            var W = Op.Shared(NN.Random.Uniform(-1f, 1f, 10, 10), "W1");
            var x1 = Op.Dot(W, x);
            x1.Name = nameof(x1);

            var y = Op.Sigmoid(Op.Tanh(x1));
            y.Name = nameof(y);

            var cost = Op.Norm2(y) + Op.Norm2(x1);

            var dW = Op.Grad(cost, W);

            var update = new OrderedDictionary { [W] = W - 0.05f * dW };

            var f = Op.Function(input: x, output: cost, updates: update);
            AssertSourceContains("Dot", exactly: 2);
        }

        [TestMethod]
        public void ComputesSigmoidOnceInsideScanOutput()
        {
            var xs = Op.Matrix<float>(-1, 10, "xs");
            var ys = Op.Scan(x => Op.Sigmoid(x), xs);
            var y = ys[-1];
            var cost = 0.5f * Op.Norm2(y);
            var dxs = Op.Grad(cost, xs);

            var f = Op.Function(input: xs, output: (cost, dxs));
            AssertSourceContains("Sigmoid", exactly: 1);
        }

        [TestMethod]
        public void ComputesSigmoidOnceInsideScanRec()
        {
            var xs = Op.Matrix<float>(-1, 10, "xs");
            var ys = Op.Scan((x, x2) => new[] { Op.Sigmoid(x) }, xs, outputsInfo: new[] { Op.Zeros<float>(10) });
            var y = ys[0][-1];
            var cost = 0.5f * Op.Norm2(y);
            // here the gradient comes from a recursive variable
            var dxs = Op.Grad(cost, xs);

            var f = Op.Function(input: xs, output: (cost, dxs));
            AssertSourceContains("Sigmoid", exactly: 1);
        }

        [TestMethod]
        public void ComputesSigmoidOnceInsideScanRec2()
        {
            var xs = Op.Matrix<float>(-1, 10, "xs");
            var ys = Op.Scan((x, x2) => new[] { x, Op.Sigmoid(x) }, xs, outputsInfo: new[] { null, Op.Zeros<float>(10) });
            var y = ys[0][-1];
            var cost = 0.5f * Op.Norm2(y);
            // here the gradient comes from a non recursive variable
            var dxs = Op.Grad(cost, xs);

            var f = Op.Function(input: xs, output: (cost, dxs));
            AssertSourceContains("Sigmoid", exactly: 1);
        }


        [TestMethod]
        public void CanCompileLoopsWithCollisionOfName()
        {
            int n = 10;
            var x0 = Op.Matrix<float>(-1, n, "x0");
            var x1 = Op.Scan((x, h) => new[] { x, x + h }, x0, new[] { null, Op.Zeros<float>(n) })[1];
            var x2 = Op.Scan((x, h) => new[] { x, x + h }, x1, new[] { null, Op.Zeros<float>(n) })[1];
            // name 'h' is used twice

            var f = Op.Function(x0, x2);
            f(NN.Random.Uniform(-1f, 1f, 5, n));
        }

        [TestMethod]
        public void CanCompileLoopUsingAxis1()
        {
            int n = 10;
            var x0 = Op.Matrix<float>(-1, n, "x0");
            var x0_T = x0.DimShuffle(1, 0);
            var x1 = Op.Scan((x, h) => new[] { x, x + h }, x0, new[] { null, Op.Zeros<float>(n) })[1];
            var x1_T = Op.Scan((x, h) => new[] { x, x + h }, x0_T, new[] { null, Op.Zeros<float>(n) }, axis: 1)[1];

            var f = Op.Function(x0, x1);
            var f_T = Op.Function(x0, x1_T);
            var _x0 = NN.Random.Uniform(-1f, 1f, 5, n);
            AssertArray.AreEqual(f(_x0), f_T(_x0));
        }

        [TestMethod]
        public void PrintPrints()
        {
            var listener = new StringTraceListener();
            Trace.Listeners.Add(listener);

            var x = Op.Scalar<string>("x");
            var print = Op.Function(input: x, output: Op.Print(x));

            var message = "Hello World";
            print(message);

            Assert.AreEqual(message + "\r\n", listener.content.ToString());
        }

        [TestMethod]
        public void PrintWithFormatPrints()
        {
            var listener = new StringTraceListener();
            Trace.Listeners.Add(listener);

            var x = Op.Scalar<float>("x");
            var print = Op.Function(input: x, output: Op.Print("Hell{0} world", x));

            print(0);

            Assert.AreEqual("Hell0 world" + "\r\n", listener.content.ToString());
        }

        private class StringTraceListener : TraceListener
        {
            public readonly StringBuilder content = new StringBuilder("");

            public override void Write(string message)
            {
                content.Append(message);
            }

            public override void WriteLine(string message)
            {
                content.AppendLine(message);
            }
        }

        private void AssertSourceContains(string substring, int exactly)
        {
            var source = FunctionBinder.Compiler.GetSource();
            var found = CountOccurences(source, substring);
            if (found != exactly)
                throw new Exception($"Found '{substring}' {found} times in generated source code, but expected exactly {exactly} times.");
        }

        private static int CountOccurences(string source, string substring)
        {
            var i = 0;
            var found = -1;
            while (i >= 0)
            {
                found += 1;
                i = source.IndexOf(substring, i);
                if (i >= 0) i += substring.Length;
            }

            return found;
        }
    }
}