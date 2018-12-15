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
using Proxem.NumNet;
using Proxem.NumNet.Single;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Samples
{
    /// <summary>
    /// Implementation of a 1D CNN.
    /// Can be use to detect pattern in a sequence of bits.
    /// It doesn't have pooling so the inputLength is fixed.
    /// </summary>
    public class SimpleCNN
    {
        // conv layer
        Tensor<float>.Shared W;
        Tensor<float>.Shared b;

        // prediction layer
        Tensor<float>.Shared S;
        Tensor<float>.Shared Sb;

        Tensor<float>.Shared[] @params;

        public Func<Array<float>, Array<float>> classify;
        public Func<Array<float>, int, float, float> train;

        /// <summary>
        ///
        /// </summary>
        /// <param name="inputLength">dimension of the input vectors</param>
        /// <param name="kernelSize">dimension of the kernel</param>
        /// <param name="nClasses">number of classes</param>
        /// <param name="scale">scaling factor for weights initialization</param>
        public SimpleCNN(int inputLength, int kernelSize, int nClasses, float scale = 0.2f)
        {
            var hiddenDim = inputLength - kernelSize + 1;

            // layers
            W = T.Shared(NN.Random.Uniform(-scale, scale, kernelSize), "W");
            b = T.Shared(NN.Zeros<float>(/*1,*/ hiddenDim), "b");

            // prediction layer
            S = T.Shared(NN.Random.Uniform(-scale, scale, nClasses, hiddenDim), "S");
            Sb = T.Shared(NN.Zeros<float>(/*1,*/ nClasses), "Sb");

            // bundle
            this.@params = new[] { W, b, S, Sb };

            var x = T.Vector<float>("x");  // [inputLength]

            var h = T.Sigmoid(T.Convolve(x, W, mode: ConvMode.Valid) + b);
            var pred = T.Softmax(T.Dot(S, h) + Sb);
            var y_pred = T.Argmax(pred, axis: 0);

            // cost and gradients and learning rate
            var y = T.Scalar<int>("y");
            var nll = -T.Mean(T.Log(pred)[y]);
            var grad = T.Grad(nll);

            var lr = T.Scalar<float>("lr");
            var updates = new OrderedDictionary();
            foreach (var W in @params)
                updates[W] = W - lr * grad[W];

            // theano functions
            this.classify = T.Function(input: x, output: pred);

            this.train = T.Function(input: (x, y, lr),
                              output: nll,
                              updates: updates);
        }
    }
}
