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

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Samples
{
    /// <summary>
    /// Implementation of a simplified LSTM. Introduced by Cho et al. in
    /// "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation"
    /// http://arxiv.org/pdf/1406.1078v3.pdf
    /// </summary>
    public class Simple        // with softmax
    {
        // layer
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
        /// <param name="inputDim">dimension of the input vectors</param>
        /// <param name="hiddenDim">dimension of the hidden layer</param>
        /// <param name="nClasses">number of classes</param>
        /// <param name="scale">scaling factor for weights initialization</param>
        public Simple(int inputDim, int hiddenDim, int nClasses, float scale = 0.2f)
        {
            // /!\ softmax requires Dot(v, M) products

            // layers
            W = T.Shared(NN.Random.Uniform(-scale, scale, inputDim, hiddenDim), "W");
            b = T.Shared(NN.Zeros<float>(hiddenDim), "b");

            // prediction layer
            S = T.Shared(NN.Random.Uniform(-scale, scale, hiddenDim, nClasses), "S");
            Sb = T.Shared(NN.Zeros<float>(nClasses), "Sb");

            // bundle
            this.@params = new[] { W, b, S, Sb };

            // Adagrad shared variables
            var hists = new Dictionary<string, Tensor<float>.Shared>();
            foreach (var param in @params)
            {
                var name = param.Name + "Hist";
                hists[name] = T.Shared(NN.Zeros<float>(param.Value.Shape), name);
            }

            // Adadelta shared variables
            var hists2 = new Dictionary<string, Tensor<float>.Shared>();
            foreach (var param in @params)
            {
                var name = param.Name + "Hist2";
                hists2[name] = T.Shared(NN.Zeros<float>(param.Value.Shape), name);
            }

            var x = T.Vector<float>("x");  // [sentence, inputDim]

            var h = T.Sigmoid(T.Dot(x, W) + b);
            var pred = T.Softmax(T.Dot(h, S) + Sb);
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
                //var grad = T.Clip(update.Item2, -10, 10);

                // Adagrad
                //const float eps = 1e-5f;
                //var hist = hists[param.Name + "Hist"];
                //updates[hist] = hist + grad * grad;
                //updates[param] = param - lr * grad / T.Sqrt(hist + eps);

                // Adadelta
                //const float rho = 0.95f;
                //const float eps = 1e-5f;
                //var hist = hists[param.Name + "Hist"];
                //var hist2 = hists2[param.Name + "Hist2"];
                //var newHist = rho * hist + (1 - rho) * (grad * grad);
                //updates[hist] = newHist;
                //var newGrad = grad * T.Sqrt((hist2 + eps) / (newHist + eps));
                //updates[param] = param - newGrad;
                //updates[hist2] = rho * hist2 + (1 - rho) * (newGrad * newGrad);

                // Regular
                updates[param] = param - lr * grad;
            }

            // theano functions
            this.classify = T.Function(input: x, output: pred);

            this.train = T.Function(input: (x, y, lr),
                              output: nll,
                              updates: updates);
        }
    }
}
