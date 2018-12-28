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
using Proxem.NumNet;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Samples
{
    public class Elman2
    {
        //FloatArrayExpr.Shared emb;
        private Tensor<float>.Shared Wx;

        private Tensor<float>.Shared Wh;
        private Tensor<float>.Shared W;
        private Tensor<float>.Shared bh;
        private Tensor<float>.Shared b;
        private Tensor<float>.Shared h0;
        private Tensor<float>.Shared[] @params;

        public readonly Func<Array<float>, Array<int>> classify;
        public readonly Func<Array<float>, int, float, float> train;

        /// <summary>
        ///
        /// </summary>
        /// <param name="nh">dimension of the hidden layer</param>
        /// <param name="nc">number of classes</param>
        /// <param name="de">dimension of the word embeddings</param>
        /// <param name="cs">word window context size</param>
        public Elman2(int nh, int nc, int de, int cs)
        {
            // parameters of the model
            this.Wx = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, de * cs, nh), "Wx");
            this.Wh = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, nh), "Wh");
            this.W = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, nc), "W");
            this.bh = T.Shared(NN.Zeros<float>(1, nh), "bh");        // /!\ shape = (1, nh)
            this.b = T.Shared(NN.Zeros<float>(1, nc), "b");          // /!\ shape = (1, nc)
            this.h0 = T.Shared(NN.Zeros<float>(1, nh), "h0");        // /!\ shape = (1, nh)

            // bundle
            this.@params = new[] { this.Wx, this.Wh, this.W, this.bh, this.b, this.h0 };

            var x = T.Tensor3<float>("x");  // [sentence, de * cs]
            var y = T.Scalar<int>("y"); // label

            Func<Tensor<float>, Tensor<float>, IList<Tensor<float>>> recurrence = (x_t, h_tm1) =>
            {
                var h_t = T.Sigmoid(T.Dot(x_t, this.Wx) + T.Dot(h_tm1, this.Wh) + this.bh);
                var s_t = T.Softmax(T.Dot(h_t, this.W) + this.b);
                return new[] { h_t, s_t };
            };

            var result = T.Scan(
                fn: recurrence,
                sequences: x,
                outputsInfo: new[] { this.h0, null }
                /*, n_steps: x.Shape[0]*/);
            var h = result[0];
            var s = result[1];

            var p_y_given_x_lastword = s[-1, 0 /*, Slicer._*/];
            var p_y_given_x_sentence = s[XSlicer._/*, 0, Slicer._*/];
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

            this.train = T.Function(input: (x, y, lr),
                              output: nll,
                              updates: updates);
        }
    }

    public class Elman3
    {
        private Tensor<float>.Shared Wx;
        private Tensor<float>.Shared Wh;
        private Tensor<float>.Shared W;
        private Tensor<float>.Shared bh;
        private Tensor<float>.Shared b;
        private Tensor<float>.Shared h0;
        private Tensor<float>.Shared[] @params;

        public readonly Func<Array<float>, Array<int>> classify;
        public readonly Func<Array<float>, int, float, float> train;
        public readonly Scalar<float> nll;

        /// <summary>
        ///
        /// </summary>
        /// <param name="nh">dimension of the hidden layer</param>
        /// <param name="nc">number of classes</param>
        /// <param name="de">dimension of the word embeddings</param>
        /// <param name="cs">word window context size</param>
        public Elman3(int nh, int nc, int de, int cs)
        {
            // parameters of the model
            var scale = 0.2f;
            this.Wx = T.Shared(scale * NN.Random.Uniform(-1.0f, 1.0f, de * cs, nh), "Wx");
            //this.Wh = T.Shared(scale * NN.Random.Uniform(-1.0f, 1.0f, nh, nh), "Wh");
            this.Wh = T.Shared(NN.Eye<float>(nh), "Wh");
            this.W = T.Shared(scale * NN.Random.Uniform(-1.0f, 1.0f, nh, nc), "W");
            this.bh = T.Shared(NN.Zeros<float>(nh), "bh");
            this.b = T.Shared(NN.Zeros<float>(nc), "b");
            this.h0 = T.Shared(NN.Zeros<float>(nh), "h0");

            // bundle
            this.@params = new[] { this.Wx, this.Wh, this.W, this.bh, this.b, this.h0 };

            var x = T.Matrix<float>("x");  // [sentence, de * cs]
            var y = T.Scalar<int>("y"); // label

            Func<Tensor<float>, Tensor<float>, Tensor<float>[]> recurrence = (x_t, h_tm1) =>
            {
                var h_t = T.Sigmoid(T.Dot(x_t, this.Wx) + T.Dot(h_tm1, this.Wh) + this.bh);
                var s_t = T.Softmax(T.Dot(h_t, this.W) + this.b);
                return new[] { h_t, s_t };
            };

            var result = T.Scan(
                fn: recurrence,
                sequences: x,
                outputsInfo: new[] { this.h0, null }
                /*, n_steps: x.Shape[0]*/);
            var h = result[0];
            var s = result[1];

            var p_y_given_x_lastword = s[-1, /*0,*/ XSlicer._];              // 0 because of Theano's Softmax ?
            var p_y_given_x_sentence = s[XSlicer._, /*0,*/ XSlicer._];
            var y_pred = T.Argmax(p_y_given_x_sentence, axis: 1);

            // cost and gradients and learning rate
            var lr = T.Scalar<float>("lr");
            nll = -T.Mean(T.Log(p_y_given_x_lastword)[y]);
            var gradients = T.Grad(nll);
            var updates = new OrderedDictionary();
            foreach (var W in @params)
                updates[W] = W - lr * gradients[W];

            // theano functions

            this.classify = T.Function(input: x, output: y_pred);

            this.train = T.Function(input: (x, y, lr),
                              output: nll,
                              updates: updates);
        }
    }
}
