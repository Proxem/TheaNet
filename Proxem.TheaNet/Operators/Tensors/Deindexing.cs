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

    /// <summary>
    /// Reverse of `Indexing`.
    /// `x = A[indices]` takes some values from `A` and stores them into `x`.
    /// `B = Deindexing(x, A.Shape, indices)` takes the values from `x`
    /// and put them back in an array with the same shape than `A` in their original positions.
    /// </summary>
    public class Deindexing<T> : Tensor<T>.NAry
    {
        public readonly Tensor<T> Content;
        public readonly TensorList Indices;

        /// <param name="x">the array to read values from</param>
        /// <param name="shape">the shape of the original array</param>
        /// <param name="indices">the indices of each values from `x`</param>
        public static Tensor<T> Create(Tensor<T> x, XList<Scalar<int>, int> shape, TensorList indices)
        {
            switch (x)
            {
                case Fill<T> fill:
                    return Dispatch<T>.Create(fill.x, shape, indices);
                default:
                    return new Deindexing<T>(x, shape, indices);
            }
        }

        private Deindexing(Tensor<T> content, XList<Scalar<int>, int> shape, TensorList indices) :
            base("Deindex", content, shape, indices)
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
            new Deindexing<T>((Tensor<T>)inputs[0], (XList<Scalar<int>, int>)inputs[1], (TensorList)inputs[2]);
    }
}
