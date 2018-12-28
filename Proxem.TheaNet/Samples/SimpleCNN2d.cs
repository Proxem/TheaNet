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
    /// Implementation of a 2D CNN.
    /// Can be use to detect pattern in a 2D sequence of bits.
    /// It doesn't have pooling so the inputLength is fixed.
    /// </summary>
    public class SimpleCNN2d
    {
        // conv layer
        Tensor<float>.Shared W;
        //Tensor<float>.Shared b;

        // prediction layer
        Tensor<float>.Shared S;
        Tensor<float>.Shared Sb;

        Tensor<float>.Shared[] @params;

        public Func<Array<float>, Array<int>> classify;
        public Func<Array<float>, Array<float>> debug;
        public Func<Array<float>, int, float, float> train;

        /// <summary>
        ///
        /// </summary>
        /// <param name="inputShape">dimension of the input vectors</param>
        /// <param name="kernelShape">dimension of the kernel</param>
        /// <param name="poolingShape">dimension of the pooling</param>
        /// <param name="nClasses">number of classes</param>
        /// <param name="scale">scaling factor for weights initialization</param>
        public SimpleCNN2d(int[] inputShape, int[] kernelShape, int[] poolingShape, int nClasses, float scale = 0.2f)
        {
            var flatShape = ((inputShape[0] - kernelShape[0] + 1) / poolingShape[0]) * ((inputShape[1] - kernelShape[1] + 1) / poolingShape[1]);
            var scaling = (float) Math.Sqrt(6f/(inputShape[0]*inputShape[1] + flatShape));
            // layers
            W = T.Shared(NN.Random.Uniform(-scaling, scaling, kernelShape), "W");
            //b = T.Shared(NN.Zeros<float>(1, 1), "b");

            //var flatShape = (inputShape[0] + kernelShape[0] - 1) * (inputShape[1] + kernelShape[1] - 1);
            // prediction layer
            var scaling2 = (float) Math.Sqrt(6f / flatShape);
            S = T.Shared(NN.Random.Uniform(-scaling2, scaling2, nClasses, flatShape), "S");
            Sb = T.Shared(NN.Zeros<float>(nClasses, 1), "Sb");

            // bundle
            this.@params = new[] { W, S, Sb };

            var x = T.Matrix<float>("x");  // [inputLength]
            var h = T.MaxPooling2d_new(T.Sigmoid(T.Convolve2d(x, W, mode: ConvMode.Valid)), poolingShape[0], poolingShape[1], true);
            //var h = T.Sigmoid(T.Convolve2d(x, W, mode: ConvMode.Full));
            Scalar<int>[] fS = new Scalar<int>[2] {flatShape, 1};
            var h2 = h.Item1().Reshape(fS);
            var debug = (T.Dot(S, h2) + Sb).Reshape(nClasses);
            var pred = T.Softmax(debug);
            var y_pred = T.Argmax(pred, axis: 0);

            // cost and gradients and learning rate
            var y = T.Scalar<int>("y");
            var nll = -T.Mean(T.Log(pred)[y]);
            var gradients = T.Grad(nll);

            var lr = T.Scalar<float>("lr");
            var updates = new OrderedDictionary();
            foreach (var param in @params)
            {
                var grad = gradients[param];
                //updates[param] = param - lr * grad;
                var rho = 1f;
                var eps = 1e-1f;
                Tensor<float>.Shared Hist = T.Shared(NN.Zeros<float>(param.Value.Shape), param.Name + "Hist");
                Tensor<float>.Shared Hist2 = T.Shared(NN.Zeros<float>(param.Value.Shape), param.Name + "Hist2");
                var newHist = rho * Hist + (1 - rho) * (grad * grad);
                updates[Hist] = newHist;
                var newGrad = grad * T.Sqrt((Hist2 + eps) / (newHist + eps));
                updates[param] = param - newGrad;
                updates[Hist2] = rho * Hist2 + (1 - rho) * (newGrad * newGrad);
            }

            // theano functions
            this.classify = T.Function(input: x, output: y_pred);
            this.debug = T.Function(input: x, output: debug);
            this.train = T.Function(input: (x, y, lr),
                              output: nll,
                              updates: updates);
        }
    }
}
