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

namespace Proxem.TheaNet.Operators.Tensors
{
    public class Max<Type> : Tensor<Type>.Aggregate
    {
        private Tensor<int> argmax;

        public Max(Tensor<Type> x, int axis) : base("Max", x, axis)
        {
        }

        public override void Backward(Tensor<Type> delta, Backpropagation bp)
        {
            argmax = argmax ?? Op.Argmax(x, Axis, keepDims: true);
            var deltaX = UnArgmax<Type>.Create(delta, argmax, Axis, x.Shape);
            bp.PushGradientTo(x, deltaX);
        }

        public override Aggregate Clone(Tensor<Type> x, int axis)
        {
            return new Max<Type>(x, Axis);
        }
    }
}
