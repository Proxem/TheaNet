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

    /// <summary>Operator for advanced indexing.</summary>
    public class Indexing<Type> : Tensor<Type>.Binary<Tensor<Type>, Array<Type>, TensorList, Array<int>[]>, IExpr<Type>
    {
        public readonly TensorList Indices;

        /// <summary>Advanced indexing</summary>
        /// <remarks>`indices.Length` should match `x.NDim`</remarks>
        /// <param name="x">the array to index</param>
        /// <param name="indices">the indices to take from `x`.</param>
        public static Tensor<Type> Create(Tensor<Type> x, Tensor<int>[] indices)
        {
            var result = new Indexing<Type>(x, indices);
            switch (x)
            {
                case Fill<Type> fill:
                    return Op.ConstLike(fill.x, result);
                default:
                    return result;
            }
        }

        private Indexing(Tensor<Type> x, TensorList indices) : base("IndexWith", x, indices)
        {
            this.Indices = indices;
            var indexDim = Indices[0].NDim;
            this.Shape = new Dim[indexDim + x.NDim - Indices.Count];

            for (int i = 1; i < Indices.Count; ++i)
                if (!Indices[i].Shape.CanEqualTo(Indices[0].Shape))
                    throw new RankException("In advanced indexing all index array must have the same size");

            // TODO : still TODO ?
            for (int i = 0; i < indexDim; ++i)
            {
                this.Shape[i] = Indices[0].Shape[i];
            }
            for (int i = NDim - 1; i >= indexDim; --i)
            {
                this.Shape[i] = x.Shape[x.NDim + i - NDim];
            }
        }

        public override Dim[] Shape { get; }

        public override void Backward(Tensor<Type> delta, Backpropagation bp) =>
            bp.PushGradientTo(x, Deindexing<Type>.Create(delta, x.Shape, Indices));

        public override Binary<Tensor<Type>, Array<Type>, TensorList, Array<int>[]> Clone(Tensor<Type> x, TensorList y) =>
            new Indexing<Type>(x, y);
    }
}
