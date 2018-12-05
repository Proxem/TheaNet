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

namespace Proxem.TheaNet.Operators.Tensors
{
    using Dim = Scalar<int>;
    using TensorList = XList<Tensor<int>, Array<int>>;
    
    /// <summary>Creates an empty array and fill some indices with a given content.</summary>
    public class Dispatch<T> : Tensor<T>.NAry
    {
        public readonly Scalar<T> Content;
        public readonly TensorList Indices;

        /// <param name="x">the scalar expression used to fill the array</param>
        /// <param name="shape">the shape of the array</param>
        /// <param name="indices">the indices to fill</param>
        public static Tensor<T> Create(Scalar<T> x, XList<Scalar<int>, int> shape, TensorList indices) =>
            new Dispatch<T>(x, shape, indices);

        private Dispatch(Scalar<T> content, XList<Scalar<int>, int> shape, TensorList indices) : base("Dispatch", content, shape, indices)
        {
            Content = content;
            Shape = shape;
            Indices = indices;
        }

        public override Dim[] Shape { get; }

        public override void Backward(Tensor<T> delta, Backpropagation bp)
        {
            throw new NotImplementedException();
        }

        public override NAry Clone(IReadOnlyList<IExpr> inputs) =>
            new Dispatch<T>((Scalar<T>)inputs[0], (XList<Scalar<int>, int>)inputs[1], (TensorList)inputs[2]);
    }
}
