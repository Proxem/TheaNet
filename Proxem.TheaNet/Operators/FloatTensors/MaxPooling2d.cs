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
using Proxem.NumNet;
using Proxem.NumNet.Single;
using Proxem.TheaNet.Binding;
using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet.Operators.FloatTensors
{
    public class MaxPooling2d : Tensor<float>.Unary<Tensor<float>, Array<float>>
    {
        private Dim[] _shape;
        public readonly int pool_h;
        public readonly int pool_w;
        public readonly bool ig;

        public MaxPooling2d(Tensor<float> x, int pool_h, int pool_w, bool ig)
            : base("NN.DownSample_MaxPooling2d", x, pool_h, pool_w)
        {
            this.pool_h = pool_h;
            this.pool_w = pool_w;
            this.ig = ig;
            _shape = new Dim[] { x.Shape[0] / pool_h, x.Shape[1] / pool_w };

            // TODO: case ignoreborder == false
            //if (ig == false)
            //{
            //    if (((x.Axes[0] ^ pool_h) >= 0) && (x.Axes[0] % pool_h != 0)) axis[0]++;
            //    if (((x.Axes[1] ^ pool_w) >= 0) && (x.Axes[1] % pool_w != 0)) axis[1]++;
            //}
        }

        public override Dim[] Shape => _shape;

        public override void Backward(Tensor<float> delta, Backpropagation bp) =>
            bp.PushGradientTo(x, new Unpooling(delta, x, pool_h, pool_w, ig));

        public override Unary<Tensor<float>, Array<float>> Clone(Tensor<float> x) => new MaxPooling2d(x, pool_h, pool_w, ig);
    }
}
