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
    public class Unpooling : Tensor<float>.Binary<Tensor<float>, Array<float>, Tensor<float>, Array<float>>
    {
        private Dim[] _shape;
        public readonly int pool_h;
        public readonly int pool_w;
        public readonly bool ig;

        public Unpooling(Tensor<float> delta, Tensor<float> x, int pool_h, int pool_w, bool ig)
            : base("NN.Unpooling", delta, x, pool_h, pool_w)
        {
            this.pool_h = pool_h;
            this.pool_w = pool_w;
            this.ig = ig;
            this._shape = x.Shape;
        }

        public override Dim[] Shape => this._shape;

        public override void Backward(Tensor<float> delta, Backpropagation bp)
        {
            throw new NotImplementedException();
        }

        public override Binary<Tensor<float>, Array<float>, Tensor<float>, Array<float>> Clone(Tensor<float> x, Tensor<float> y) =>
            new Unpooling(x, y, pool_h, pool_w, ig);
    }
}
