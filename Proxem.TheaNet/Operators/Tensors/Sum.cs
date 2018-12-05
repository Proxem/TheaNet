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

namespace Proxem.TheaNet.Operators.Tensors
{
    public class Sum<Type> : Tensor<Type>.Aggregate
    {
        static public Tensor<Type> Create(Tensor<Type> x, int axis)
        {
            axis = axis < 0 ? axis + x.NDim : axis;
            var result = new Sum<Type>(x, axis);

            switch (x)
            {
                case Fill<Type> fill:
                    return Op.ConstLike(fill.x * x.Shape[axis].As<Type>(), result);
                case OneHot<Type> oneHot:
                    return axis == 0 ? oneHot.Content.Reshape(result.Shape) : result;
                case OneHotPoint<Type> oneHotPoint:
                    var point = ((Scalar<int>[])oneHotPoint.Indexes).ToArray();
                    point[axis] = 0;
                    return Op.OneHot(result.Shape, point, oneHotPoint.Content);
                default:
                    return result;
            }
        }

        private Sum(Tensor<Type> x, int axis) : base("Sum", x, axis)
        {
        }

        public override void Backward(Tensor<Type> delta, Backpropagation bp)
        {
            bp.PushGradientTo(x, delta.BroadcastTo(x.Shape));
        }

        public override Aggregate Clone(Tensor<Type> x, int axis)
        {
            return new Sum<Type>(x, axis);
        }
    }
}
