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
    /// Implementation of a simplified LSTM, know as GRU (Gated Recurrent Unit).
    /// Introduced by Cho et al. in
    /// <a href="http://arxiv.org/pdf/1406.1078v3.pdf">
    /// "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation"
    /// </a>
    ///
    /// For classical LSTM, see <a href="http://arxiv.org/pdf/1308.0850v5.pdf">this paper</a>.
    /// Uses Softmax
    /// </summary>
    public class GRU2        // with softmax
    {
        // initial hidden state
        Tensor<float>.Shared h0;

        // reset gate layers
        Tensor<float>.Shared Wr;
        Tensor<float>.Shared Ur;
        Tensor<float>.Shared br;

        // update gate layers
        Tensor<float>.Shared Wz;
        Tensor<float>.Shared Uz;
        Tensor<float>.Shared bz;

        // layer
        Tensor<float>.Shared W;
        Tensor<float>.Shared U;
        Tensor<float>.Shared b;

        // prediction layer
        Tensor<float>.Shared S;
        Tensor<float>.Shared Sb;

        Tensor<float>.Shared[] @params;

        public Func<Array<float>, Array<int>> classify;
        public Func<Array<float>, int, float, float> train;

        /// <summary></summary>
        /// <param name="inputDim">dimension of the input vectors</param>
        /// <param name="hiddenDim">dimension of the hidden layer</param>
        /// <param name="nClasses">dimension of the output vector</param>
        /// <param name="scale">scaling factor to initialize weights</param>
        public GRU2(int inputDim, int hiddenDim, int nClasses, float scale = 0.2f)
        {
            // /!\ softmax requires Dot(v, M) products

            // initial hidden state
            h0 = T.Shared(NN.Zeros<float>(hiddenDim), "h0");

            // reset gate layers
            Wr = T.Shared(NN.Random.Uniform(-scale, scale, inputDim, hiddenDim), "Wr");
            Ur = T.Shared(NN.Eye<float>(hiddenDim), "Ur");
            br = T.Shared(NN.Zeros<float>(/*1,*/ hiddenDim), "br");

            // update gate layers
            Wz = T.Shared(NN.Random.Uniform(-scale, scale, inputDim, hiddenDim), "Wz");
            Uz = T.Shared(NN.Eye<float>(hiddenDim), "Uz");
            bz = T.Shared(NN.Zeros<float>(/*1,*/ hiddenDim), "bz");

            // layers
            W = T.Shared(NN.Random.Uniform(-scale, scale, inputDim, hiddenDim), "W");
            U = T.Shared(NN.Eye<float>(hiddenDim), "U");
            b = T.Shared(NN.Zeros<float>(/*1,*/ hiddenDim), "b");

            // prediction layer
            S = T.Shared(NN.Random.Uniform(-scale, scale, hiddenDim, nClasses), "S");
            Sb = T.Shared(NN.Zeros<float>(/*1,*/ nClasses), "Sb");

            // bundle
            this.@params = new[] { h0, Wr, Ur, br, Wz, Uz, bz, W, U, b, S, Sb };

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

            var x = T.Matrix<float>("x");  // [sentence, inputDim]
            var y = T.Scalar<int>("y");

            Func<Tensor<float>, Tensor<float>, Tensor<float>[]> recurrence = (x_t, h_tm1) =>
            {
                // reset gate
                var r_t = T.Sigmoid(T.Dot(x_t, Wr) + T.Dot(h_tm1, Ur) + br);
                // update gate
                var z_t = T.Sigmoid(T.Dot(x_t, Wz) + T.Dot(h_tm1, Uz) + bz);
                // proposed hidden state
                var _h_t = T.Tanh(T.Dot(x_t, W) + T.Dot(r_t * h_tm1, U) + b);
                // actual hidden state
                var h_t = z_t * h_tm1 + (1 - z_t) * _h_t;
                // return all the intermediate variables because they may be reused by T.Grad to optimize gradient computation
                return new[] { h_t, r_t, z_t, _h_t };
            };

            var h = T.Scan(recurrence, x, new[] { h0, null, null, null })[0][-1];
            var pred = T.Softmax(T.Dot(h, S) + Sb);
            var y_pred = T.Argmax(pred, axis: 0);

            // cost and gradients and learning rate
            var lr = T.Scalar<float>("lr");
            var nll = -T.Mean(T.Log(pred)[y]);
            var gradients = T.Grad(nll);

            var updates = new OrderedDictionary();
            foreach (var param in @params)
            {
                var grad = gradients[param];
                //var grad = T.Clip(update.Item2, -10, 10);

                // Adagrad
                const float eps = 1e-5f;
                var hist = hists[param.Name + "Hist"];
                updates[hist] = hist + grad * grad;
                updates[param] = param - lr * grad / T.Sqrt(hist + eps);

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
                //updates[param] = param - lr * grad;
            }

            // theano functions
            this.classify = T.Function(input: x, output: y_pred);

            this.train = T.Function(input1: x, input2: y, input3: lr,
                              output: nll,
                              updates: updates);
        }
    }
}
