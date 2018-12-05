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

using Proxem.NumNet;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Samples
{
    public static class UpdateRules
    {
        // usage: var train = T.Function(...vars..., output: loss, updates: UpdateRules.Sgd(loss, eta, @params));
        public static OrderedDictionary Sgd(Scalar<float> loss, Scalar<float> lr, params Tensor<float>.Shared[] @params)
        {
            var dloss = T.Grad(loss);
            var result = new OrderedDictionary();
            foreach (var param in @params)
            {
                result[param] = param - lr * dloss[param];
            }
            return result;
        }

        // usage: var train = T.Function(...vars..., output: loss, updates: UpdateRules.Adam(loss, @params));
        public static OrderedDictionary Adam(Scalar<float> loss, Tensor<float>.Shared[] @params,
            float lr = 0.001f,      // alpha
            float b1 = 0.9f,
            float b2 = 0.999f,
            float eps = 1e-8f,
            float lambda = 1 - 1e-8f)
        {
            // Default values are taken from [Kingma2014]

            // [Kingma2014] Kingma, Diederik, and Jimmy Ba. "Adam: A Method for Stochastic Optimization."
            // http://arxiv.org/pdf/1412.6980v4.pdf

            var result = new OrderedDictionary();

            var t = T.Shared<float>(1, "t");
            var b1t = b1 * T.Pow(lambda, (Scalar<float>)t);     // decay first moment running average coefficient
            var mc = 1 - T.Pow(b1, (Scalar<float>)(t + 1));     // first moment bias correction
            var vc = 1 - T.Pow(b2, (Scalar<float>)(t + 1));     // second raw moment bias correction
            result[t] = t + 1;

            var dloss = T.Grad(loss);
            foreach (var param in @params)
            {
                var m = T.Shared(NN.Zeros(param.Value.Shape), "m_" + param.Name);
                var v = T.Shared(NN.Zeros(param.Value.Shape), "v_" + param.Name);

                var g = dloss[param];
                var mt = b1t * m + (1 - b1t) * g;               // moving average of first moment (mean)
                var vt = b2 * v + (1 - b2) * T.Pow(g, 2);       // moving average of second raw moment (uncentered variance)
#if true
                var mh = mt / mc;                               // bias-corrected first moment
                var vh = vt / vc;                               // bias-corrected second raw moment
                var paramt = param - (lr * mh) / (T.Sqrt(vh) + eps);
#else
                var alpha = lr * T.Sqrt(vc) / mc;
                var paramt = param - (alpha * mt) / (T.Sqrt(vt) + eps);
#endif

                result[m] = mt;
                result[v] = vt;
                result[param] = paramt;
            }
            return result;
        }
    }
}
