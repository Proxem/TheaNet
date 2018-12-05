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

using Proxem.NumNet;
using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet.Operators.FloatTensors
{
    public class Softmax : Tensor<float>.Unary<Tensor<float>, Array<float>>
    {
        public readonly int Axis;

        public static Tensor<float> Create(Tensor<float> x, int axis = -1)
        {
            switch (x)
            {
                case BroadCast<float> xBroadcast:
                    if (xBroadcast.broadcast.Contains(axis))
                        return Op.Const(1f / xBroadcast.Shape[axis].As<float>(), xBroadcast.Shape);
                    else
                        return Op.Softmax(xBroadcast.x, axis).BroadcastTo(xBroadcast.Shape);
                default:
                    return new Softmax(x, axis);
            }
        }

        private Softmax(Tensor<float> x, int axis = -1) : base("Softmax", x, axis.Named("axis"))
        {
            Axis = axis;
        }

        public override Dim[] Shape => x.Shape;

        public override void Backward(Tensor<float> delta, Backpropagation bp)
        {
            var tmp = delta * this;
            bp.PushGradientTo(x, tmp - this * Op.Sum(tmp, axis: Axis, keepDims: true));
        }

        public override Unary<Tensor<float>, Array<float>> Clone(Tensor<float> x)
        {
            return new Softmax(x, Axis);
        }
    }
}
