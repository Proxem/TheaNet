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
using System.Globalization;
using System.Linq;
using Proxem.NumNet;
using Proxem.TheaNet;
using Proxem.TheaNet.Binding;

using Dim = Proxem.TheaNet.Scalar<int>;


namespace Proxem.TheaNet.Operators.Tensors
{
    /// <summary>
    /// Sets only one point in an array.
    /// `x = OneHotPoint(shape, [index0, index1], content);` is equivalent to:
    /// `x = Zeros(shape); x.Item[index0, index1] = content;`
    /// </summary>
    public class OneHotPoint<Type> : Tensor<Type>.NAry
    {
        public static Tensor<Type> Create(Dim[] shape, Scalar<int>[] indexes, Scalar<Type> content = null)
        {
            content = content ?? Numeric<Type>.One;
            if (shape.Length == 1 && shape[0].IsOne)
                return Op.Const(content, shape);

            if (shape.Length != indexes.Length) throw new ArgumentException();
            return new OneHotPoint<Type>(shape, indexes, content);
        }

        private OneHotPoint(XList<Scalar<int>, int> shape, XList<Scalar<int>, int> indexes, Scalar<Type> content)
            : base("OneHotPoint", shape, indexes, content)
        {
        }

        public override Dim[] Shape => (XList<Scalar<int>, int>)Inputs[0];

        public XList<Scalar<int>, int> Indexes => (XList<Scalar<int>, int>)Inputs[1];

        public Scalar<Type> Content => (Scalar<Type>)Inputs[2];

        public override void Backward(Tensor<Type> delta, Backpropagation bp) =>
            bp.PushGradientTo(this.Content, delta.Item[this.Indexes]);

        public override NAry Clone(IReadOnlyList<IExpr> inputs) =>
            new OneHotPoint<Type>((XList<Scalar<int>, int>)inputs[0], (XList<Scalar<int>, int>)inputs[1], (Scalar<Type>)inputs[2]);
    }
}
