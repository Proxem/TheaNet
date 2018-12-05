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
    /// <summary>
    /// Sets only a part of an array.
    /// `x = OneHot(shape, index, content);` is equivalent to:
    /// `x = Zeros(shape); x[index, :] = content;`
    /// </summary>
    public class OneHot<Type> : Tensor<Type>.NAry
    {
        public static Tensor<Type> Create(Dim[] shape, Scalar<int> index, Tensor<Type> content = null)
        {
            content = content ?? Op.Ones<Type>(shape.DropLeft(1));
            return new OneHot<Type>(shape, index, content);
        }

        private OneHot(XList<Scalar<int>, int> shape, Scalar<int> index, Tensor<Type> content) : base("OneHot", shape, index, content)
        {
        }

        public Tensor<Type> Content => (Tensor<Type>)this.Inputs.Skip(2).First();

        public Scalar<int> Index => (Scalar<int>)this.Inputs.Skip(1).First();

        public override Dim[] Shape => (XList<Scalar<int>, int>)this.Inputs.First();

        public override void Backward(Tensor<Type> delta, Backpropagation bp) =>
            bp.PushGradientTo(Content, delta[this.Index]);

        public override NAry Clone(IReadOnlyList<IExpr> inputs) =>
            new OneHot<Type>((XList<Scalar<int>, int>)inputs[0], (Scalar<int>)inputs[1], (Tensor<Type>)inputs[2]);
    }
}
