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
using Proxem.TheaNet.Operators.FloatTensors;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Samples
{

    public class Elman
    {
        Tensor<float>.Shared emb, Wx, Wh, W, bh, b, h0;
        public readonly Scalar<float> Loss;
        public readonly Tensor<float>.Shared[] @params;
        string[] names;

        public Func<Array<int>, Array<int>> classify;
        public Func<Array<int>, int, float, float> train;
        public Action normalize;

        /// <summary>
        ///
        /// </summary>
        /// <param name="nh">dimension of the hidden layer</param>
        /// <param name="nc">number of classes</param>
        /// <param name="ne">number of word embeddings in the vocabulary</param>
        /// <param name="de">dimension of the word embeddings</param>
        /// <param name="cs">word window context size</param>
        public Elman(int nh, int nc, int ne, int de, int cs)
        {
            // parameters of the model
            this.emb = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, ne + 1, de), "emb"); // add one for PADDING at the end
            this.Wx = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, de * cs, nh), "Wx");
            this.Wh = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, nh), "Wh");
            this.W = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, nc), "W");
            this.bh = T.Shared(NN.Zeros<float>(nh), "bh");
            this.b = T.Shared(NN.Zeros<float>(nc), "b");
            this.h0 = T.Shared(NN.Zeros<float>(nh), "h0");

            // bundle
            this.@params = new[] { this.emb, this.Wx, this.Wh, this.W, this.bh, this.b, this.h0 };
            this.names = new[] { "embeddings", "Wx", "Wh", "W", "bh", "b", "h0" };
            var idxs = T.Matrix<int>("idxs"); // as many columns as context window size/lines as words in the sentence
            var x = this.emb[idxs].Reshape(idxs.Shape[0], de * cs);
            // joc: idxs.shape = [sentence, cs], emb.shape = [ne, de], emb[idx].shape = [sentence, cs, de], reshape = [sentence, de * cs]
            var y = T.Scalar<int>("y"); // label

            Func<Tensor<float>, Tensor<float>, Tensor<float>[]> recurrence = (x_t, h_tm1) =>
            {
                var h_t = T.Sigmoid(T.Dot(x_t, this.Wx) + T.Dot(h_tm1, this.Wh) + this.bh);
                var s_t = T.Softmax(T.Dot(h_t, this.W) + this.b);
                return new[] { h_t, s_t };
            };

            var result = T.Scan(fn: recurrence,
                sequences: x, outputsInfo: new[] { this.h0, null }
                /*, n_steps: x.Shape[0]*/);
            var h = result[0];
            var s = result[1];

            var p_y_given_x_lastword = s[-1, /*0,*/ XSlicer._];              // 0 because of Theano's Softmax ?
            var p_y_given_x_sentence = s[XSlicer._, /*0,*/ XSlicer._];
            var y_pred = T.Argmax(p_y_given_x_sentence, axis: 1);

            // cost and gradients and learning rate
            var lr = T.Scalar<float>("lr");
            Loss = -T.Mean(T.Log(p_y_given_x_lastword)[y]);
            var gradients = T.Grad(Loss);
            var updates = new OrderedDictionary();
            foreach (var W in @params)
                updates[W] = W - lr * gradients[W];

            // theano functions
            this.classify = T.Function(input: idxs, output: y_pred);

            this.train = T.Function(input: (idxs, y, lr),
                              output: Loss,
                              updates: updates);

            this.normalize = T.Function(updates: new OrderedDictionary {
                { emb, emb / T.Sqrt(T.Sum(T.Pow(emb, 2), axis: 1)).DimShuffle(0, 'x') }
            });
        }
    }
}
