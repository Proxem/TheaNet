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
using System.Linq;

using Proxem.NumNet;
using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet
{
    /// <summary>Broadcast a tensor to another shape.</summary>
    /// <remarks>This operator is generally simplified before code generation, if not raise an issue.</remarks>
    public class BroadCast<Type> : Tensor<Type>.Binary<Tensor<Type>, Array<Type>, XList<Scalar<int>, int>, int[]>
    {
        public readonly int[] broadcast;

        public static Tensor<Type> Create(Tensor<Type> x, Dim[] shape)
        {
            x.AssertOfDim(shape.Length);
            // we broadcast along axis i only if (x.Shape[i] == 1 && shape[i] != 1)
            var broadcast = x.Shape
                .Select(a => a as Dim.Const).Where(a => a != null && a.Value == 1)
                .Select((_, i) => shape[i] as Dim.Const).Where(a => a == null || a.Value != 1)
                .Select((a, i) => i).ToArray();

            if (broadcast.Length == 0) return x;
            else return new BroadCast<Type>(x, shape, broadcast);
        }

        private BroadCast(Tensor<Type> x, XList<Scalar<int>, int> shape, int[] broadcast): base("Broadcast", x, shape)
        {
            this.broadcast = broadcast;
        }

        public override Scalar<int>[] Shape => (XList<Scalar<int>, int>)Inputs[1];

        public override void Backward(Tensor<Type> delta, Backpropagation bp)
        {
            if (delta.IsZero) return;
            foreach (var a in broadcast)
                delta = Op.Sum(delta, axis: a, keepDims: true);
            bp.PushGradientTo(x, delta);
        }

        public override Binary<Tensor<Type>, Array<Type>, XList<Scalar<int>, int>, int[]> Clone(Tensor<Type> x, XList<Scalar<int>, int> y) =>
            new BroadCast<Type>(x, y, this.broadcast);
    }
}
