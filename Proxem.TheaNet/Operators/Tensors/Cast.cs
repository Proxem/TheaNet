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
using Proxem.TheaNet.Binding;

using Dim = Proxem.TheaNet.Scalar<int>;


namespace Proxem.TheaNet.Operators.Tensors
{
    // TODO: remove this class,
    // if `Elementwise` was less typed, it could be an `Elementwise`
    
    /// <summary>Casts an array from one type to another.</summary>
    public class Cast<T, U> : Tensor<T>.Unary<Tensor<U>, Array<U>>
    {
        public Cast(Tensor<U> x) : base("CastTensor", x) {}

        public override Dim[] Shape => x.Shape;

        public override void Backward(Tensor<T> delta, Backpropagation bp)
        {
            bp.PushGradientTo(x, delta.As<U>());
        }

        public override Unary<Tensor<U>, Array<U>> Clone(Tensor<U> x)
        {
            return new Cast<T, U>(x);
        }
    }
}
