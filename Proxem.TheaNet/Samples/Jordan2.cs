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
    public class Jordan2
    {
        //FloatArrayExpr.Shared emb;
        Tensor<float>.Shared Wx;
        Tensor<float>.Shared Ws;
        Tensor<float>.Shared W;
        Tensor<float>.Shared bh;
        Tensor<float>.Shared b;
        Tensor<float>.Shared s0;
        Tensor<float>.Shared[] @params;

        public Func<Array<float>, Array<int>> classify;
        public Func<Array<float>, int, float, float> train;

        /// <summary>
        ///
        /// </summary>
        /// <param name="nh">dimension of the hidden layer</param>
        /// <param name="nc">number of classes</param>
        /// <param name="de">dimension of the word embeddings</param>
        /// <param name="cs">word window context size</param>
        public Jordan2(int nh, int nc, int de, int cs)
        {
            var scale = 0.2f;
            // parameters of the model
            this.Wx = T.Shared(scale * NN.Random.Uniform(-1.0f, 1.0f, de * cs, nh), "Wx");
            this.Ws = T.Shared(scale * NN.Random.Uniform(-1.0f, 1.0f, nc, nh), "Wh");
            this.W = T.Shared(scale * NN.Random.Uniform(-1.0f, 1.0f, nh, nc), "W");
            this.bh = T.Shared(NN.Zeros<float>(nh), "bh");
            this.b = T.Shared(NN.Zeros<float>(nc), "b");
            this.s0 = T.Shared(NN.Zeros<float>(nc), "s0");

            // bundle
            this.@params = new[] { this.Wx, this.Ws, this.W, this.bh, this.b, this.s0 };

            var x = T.Matrix<float>("x");  // [sentence, de * cs]
            var y = T.Scalar<int>("y"); // label

            Func<Tensor<float>, Tensor<float>, IList<Tensor<float>>> recurrence = (x_t, s_tm1) =>
            {
                var h_t = T.Sigmoid(T.Dot(x_t, this.Wx) + T.Dot(s_tm1, this.Ws) + this.bh);
                var s_t = T.Softmax(T.Dot(h_t, this.W) + this.b)/*[0]*/;         // theano's softmax is 2D => 2D, event if input is 1D
                return new[] { h_t, s_t };
            };

            var result = T.Scan(fn: recurrence,
                sequences: x, outputsInfo: new[] { null, this.s0 }
                /*, n_steps: x.Shape[0]*/);
            var h = result[0];
            var s = result[1];

            var p_y_given_x_lastword = s[-1, XSlicer._];
            var p_y_given_x_sentence = s;
            var y_pred = T.Argmax(p_y_given_x_sentence, axis: 1);

            // cost and gradients and learning rate
            var lr = T.Scalar<float>("lr");
            var nll = -T.Mean(T.Log(p_y_given_x_lastword)[y]);
            var gradients = T.Grad(nll);
            var updates = new OrderedDictionary();
            foreach (var W in @params)
                updates[W] = W - lr * gradients[W];

            // theano functions
            this.classify = T.Function(input: x, output: y_pred);

            this.train = T.Function(input1: x, input2: y, input3: lr,
                              output: nll,
                              updates: updates);
        }
    }
}
