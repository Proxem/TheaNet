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

namespace Proxem.TheaNet.Test
{

    [TestClass]
    public class TestSkipGram
    {
        [TestInitialize]
        public void Initialize()
        {
            Runtime.Reset();
        }

        [TestMethod]
        public void SkipGram()
        {
            var x = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100), "x");
            var y = T.Scalar<int>("y");

            var Wout = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100, 5000), "Wout");

            var y_pred = T.Softmax(T.Dot(x, Wout)).Named("y_pred");
            var loss = -T.Log(y_pred).Item[y].Named("loss");

            var grad = T.Grad(loss);

            var fin = T.Function(y, grad[x]);
            var fout = T.Function(y, grad[Wout]);
            fin(13);
            fout(13);
        }

        [TestMethod]
        public void SkipGramNs()
        {
            var x = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100), "x");

            var Wout = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100), "Wout");

            var y_pred = T.Sigmoid((Scalar<float>)T.Dot(x, Wout)).Named("y_pred");      // TODO: Operator ScalarDot
            var loss = -T.Log(y_pred).Named("loss");

            var grad = T.Grad(loss);

            var fx = T.Function(grad[x]);
            var fout = T.Function(grad[Wout]);
            fx();
            fout();
        }


        [TestMethod] 
        public void AdaGramLike()
        {
            //var x = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100), "x");
            var s = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100, 5), "s");          // 5 sense vectors
            var alpha = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 5), "alpha");       // 5 real coefficient => softmax gives 5 probabilistic coefficient
            var p = T.Softmax(alpha).Named("p");
            var x = T.Dot(s, p).Named("x");                          // x = vector of size 100 = word vector

            var y = T.Scalar<int>("y");

            var Wout = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100, 5000), "Wout");

            var y_pred = T.Softmax(T.Dot(x, Wout)).Named("y_pred");
            var loss = -T.Log(y_pred).Item[y].Named("loss");

            var grad = T.Grad(loss);

            var fs = T.Function(y, grad[s]);                    // fs = dy/ds
            var falpha = T.Function(y, grad[alpha]);            // falpha = dy/dalpha
            var fout = T.Function(y, grad[Wout]);               // fout = dy/dWout

            fs(13);
            falpha(13);
            fout(13);
        }

        [TestMethod]
        public void AdaGramLikeNs()
        {
            //var x = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100), "x");
            var s = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100, 5), "s");          // 5 sense vectors
            var alpha = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 5), "alpha");       // 5 real coefficient => softmax gives 5 probabilistic coefficient
            var p = T.Softmax(alpha).Named("p");
            var x = T.Dot(s, p).Named("x");                          // x = vector of size 100 = word vector

            var Wout = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100), "Wout");

            var y_pred = T.Sigmoid((Scalar<float>)T.Dot(x, Wout)).Named("y_pred");
            var loss = -T.Log(y_pred).Named("loss");

            var grad = T.Grad(loss);

            var fs = T.Function(grad[s]);                    // fs = dy/ds
            var falpha = T.Function(grad[alpha]);            // falpha = dy/dalpha
            var fout = T.Function(grad[Wout]);               // fout = dy/dWout

            fs();
            falpha();
            fout();
        }

        [TestMethod]
        public void SkipGramWithBias()
        {
            var x = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100), "x");
            var bin = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100), "bin");
            var y = T.Scalar<int>("y");

            var bout = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 5000), "b");
            var Wout = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100, 5000), "Wout");

            var y_pred = T.Softmax(T.Dot(x + bin, Wout) + bout);
            var loss = -T.Log(y_pred).Item[y];

            var grad = T.Grad(loss);

            var fin = T.Function(y, grad[x]);
            var fbin = T.Function(y, grad[bin]);
            var fout = T.Function(y, grad[Wout]);
            var fbout = T.Function(y, grad[bout]);

        }

        [TestMethod]
        public void SkipGramWithRegularization()
        {
            var x = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100), "x");
            var y = T.Scalar<int>("y");

            var Wout = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100, 5000), "Wout");

            var xortho = T.Vector<float>("xortho");
            var l1 = (Scalar<float>)T.Dot(x, xortho, transposeX: true);

            var y_pred = T.Softmax(T.Dot(x, Wout));
            var loss = -T.Log(y_pred).Item[y] + 0.5f * l1 * l1;

            var grad = T.Grad(loss);

            var fin = T.Function(xortho, y, grad[x]);
            var fout = T.Function(xortho, y, grad[Wout]);
            var xortho_ = NN.Random.Uniform(-1f, 1f, 100);
            fin(xortho_, 10);
            fout(xortho_, 20);
        }

        [TestMethod]
        public void SkipGramWithRegularization2()
        {
            var x = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100), "x");
            var y = T.Scalar<int>("y");

            var Wout = T.Shared(NN.Random.Uniform(-1.0f, 1.0f, 100, 5000), "Wout");

            var xortho = T.Vector<float>("xortho");
            var l1 = (Scalar<float>)T.Dot(x, xortho, transposeX: true);

            var y_pred = T.Softmax(T.Dot(x, Wout));
            var loss = -T.Log(y_pred).Item[y] + 0.5f * l1 * l1;

            var grad = T.Grad(loss);

            var fin = T.Function(xortho, y, grad[x]);
            var fout = T.Function(xortho, y, grad[Wout]);
            var xortho_ = NN.Random.Uniform(-1f, 1f, 100);
            fin(xortho_, 10);
            fout(xortho_, 20);
        }
    }
}
