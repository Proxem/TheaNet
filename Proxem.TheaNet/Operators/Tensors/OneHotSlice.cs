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

using Slice = Proxem.NumNet.Slice;
using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet.Operators.Tensors
{
    /// <summary>
    /// Sets only a slice of an array.
    /// `x = OneHotSlice(shape, slice, content);` is equivalent to:
    /// `x = Zeros(shape); x[slice] = content;`
    /// </summary>
    /// <remarks>Similar to `OneHot` but more generic, and therefore harder to simplify.</remarks>
    public class OneHotSlice<T> : Tensor<T>.NAry
    {
        public static Tensor<T> Create(Dim[] shape, XList<XSlice, Slice> slices, Tensor<T> content)
        {
            if(slices[0].IsSingleton)
                if(slices.Values.Skip(1).All(s => s == XSlicer._))
                   return Op.OneHot(shape, slices[0].Start, content);
            return new OneHotSlice<T>(shape, slices, content);
        }

        private OneHotSlice(XList<Scalar<int>, int> shape, XList<XSlice, Slice> slices, Tensor<T> content)
            : base("OneHot", shape, slices, content)
        {
        }

        public override Dim[] Shape => (XList<Scalar<int>, int>)Inputs[0];

        public XList<XSlice, Slice> Slices => (XList<XSlice, Slice>)Inputs[1];

        public Tensor<T> Content => (Tensor<T>)Inputs[2];

        public override void Backward(Tensor<T> delta, Backpropagation bp) =>
            bp.PushGradientTo(Content, delta[Slices]);

        public override NAry Clone(IReadOnlyList<IExpr> inputs) =>
            new OneHotSlice<T>((XList<Scalar<int>, int>)inputs[0], (XList<XSlice, Slice>)inputs[1], (Tensor<T>)inputs[2]);
    }
}
