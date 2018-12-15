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
    public class Rnn
    {
        Tensor<float>.Shared Wbit;
        Tensor<float>.Shared Wstate;
        Tensor<float>.Shared Wout;
        Tensor<float>.Shared state0;
        Tensor<float>.Shared[] @params;

        public Func<Array<float>, Array<float>> classify;
        public Func<Array<float>, Array<float>, float, float> train;
        public Func<Array<float>, Array<float>, Array<float>, float> train2;

        /// <summary>
        ///
        /// </summary>
        /// <param name="nh">dimension of the hidden layer</param>
        public Rnn(int nh)
        {
            // parameters of the model
            this.Wbit = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, 1), "Wbit");
            this.Wstate = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, nh, nh), "Wstate");
            this.Wout = T.Shared(0.2f * NN.Random.Uniform(-1.0f, 1.0f, 1, nh), "Wout");
            this.state0 = T.Shared(NN.Zeros<float>(nh, 1), "state0");

            // bundle
            this.@params = new[] { this.Wbit, this.Wstate, this.Wout };     // temp: bug in computing h0

            var bits = T.Matrix<float>("bits");                // n x 1
            var expected = T.Matrix<float>("expected");        // 1 x 1

            Func<Tensor<float>, Tensor<float>, Tensor<float>> recurrence = (bit, oldState) =>
            {
                return T.Tanh(T.Dot(this.Wbit, bit) + T.Dot(this.Wstate, oldState));
            };

            var states = T.Scan(fn: recurrence,
                sequence: bits, outputsInfo: this.state0
                /*, n_steps: x.Shape[0]*/);

            //var output = T.Sigmoid(T.Dot(this.Wout, states[-1]));
            var output = T.Dot(this.Wout, states[-1]);
            var error = 0.5f * T.Norm2(output - expected);

            this.classify = T.Function(bits, output);

            var lr = T.Scalar<float>("lr");
            var gradients = T.Grad(error);
            var updates = new OrderedDictionary();
            foreach (var W in @params)
                updates[W] = W - lr * gradients[W];

            // theano functions
            this.train = T.Function(input: (bits, expected, lr),
                              output: error,
                              updates: updates);

            var bit1 = T.Matrix<float>("bit1");
            var bit2 = T.Matrix<float>("bit2");
            var test = recurrence(bit2, recurrence(bit1, state0));
            test = T.Dot(this.Wout, test);
            //test = T.Sigmoid(T.Dot(this.Wout, test));
            var e = 0.5f * T.Norm2(test - expected);

            var g = T.Grad(e);
            var updates2 = new OrderedDictionary();
            foreach (var W in @params)
                updates2[W] = W - 0.001f * gradients[W];

            this.train2 = T.Function(input: (bit1, bit2, expected), output: e, updates: updates2);
        }
    }
}
