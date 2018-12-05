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

using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet.Operators.Tensors
{
    /// <summary>Fills an empty array with the given value.</summary>
    public class Fill<Type> : Tensor<Type>.Binary<Scalar<Type>, Type, XList<Scalar<int>, int>, int[]>
    {
        /// <param name="x">the scalar expression used to fill the array</param>
        /// <param name="shape">the shape of the array</param>
        public static Tensor<Type> Create(Scalar<Type> x, XList<Scalar<int>, int> shape) => new Fill<Type>(x, shape);

        private Fill(Scalar<Type> x, Dim[] shape): base("Const", x, shape)
        {
        }

        public override Dim[] Shape => (XList<Scalar<int>, int>)this.Inputs.Skip(1).First();

        public override void Backward(Tensor<Type> delta, Backpropagation bp) =>
            bp.PushGradientTo(x, Op.Sum(delta));

        public override Binary<Scalar<Type>, Type, XList<Scalar<int>, int>, int[]> Clone(Scalar<Type> x, XList<Scalar<int>, int> y) =>
            new Fill<Type>(x, y);
    }
}
