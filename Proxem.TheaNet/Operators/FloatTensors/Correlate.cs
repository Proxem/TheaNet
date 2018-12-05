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

using Proxem.TheaNet.Binding;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Proxem.NumNet;
using Proxem.NumNet.Single;

using static Proxem.TheaNet.Operators.FloatTensors.Convolve;
using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet.Operators.FloatTensors
{
    public class Correlate : Tensor<float>.Binary<Tensor<float>, Array<float>, Tensor<float>, Array<float>>
    {
        public readonly ConvMode mode;
        private Dim[] _shape;

        public Correlate(Tensor<float> x, Tensor<float> kernel, ConvMode mode = ConvMode.Valid) :
            base("Correlate", x, kernel, mode.Named("mode"))
        {
            if (x.NDim != 1 && kernel.NDim != 1) throw new RankException("Expect inputs of dim 1");
            this.mode = mode;
            _shape = new[] { GetConvolveDim(x.Shape[0], kernel.Shape[0], mode) };
        }

        public override Dim[] Shape => _shape;

        public override void Backward(Tensor<float> delta, Backpropagation bp)
        {
            bp.PushGradientTo(x, new Convolve(delta, y, Reverse(mode)));
            bp.PushGradientTo(y, new Convolve(x, delta, Reverse(mode)));
        }

        public override Binary<Tensor<float>, Array<float>, Tensor<float>, Array<float>> Clone(Tensor<float> x, Tensor<float> y) =>
            new Correlate(x, y, mode);
    }
}
