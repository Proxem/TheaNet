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

using Proxem.TheaNet.Operators.Tensors;

namespace Proxem.TheaNet.Operators.Scalars
{
    /// <summary>Extracts an element from a Tensor.</summary>
    public class Item<Type> : Scalar<Type>.NAry
    {
        public Tensor<Type> x => (Tensor<Type>)this.Inputs.First();
        public XList<Scalar<int>, int> Indexes => (XList<Scalar<int>, int>)this.Inputs.Skip(1).First();

        private static Dictionary<string, Scalar<Type>> Cache = new Dictionary<string, Scalar<Type>>();
        public static Scalar<Type> Create(Tensor<Type> x, params Scalar<int>[] indexes)
        {
            var key = $"Item|{x.Id}|{string.Join("|", indexes.Select(index => index.Id.ToString()))}";
            if (Cache.TryGetValue(key, out var result)) return result;

            x.AssertOfDim(indexes.Length);
            switch (x)
            {
                case Fill<Type> fill:
                    result = fill.x;
                    break;
                case OneHotPoint<Type> ohp when ohp.Indexes.SequenceEqual(indexes):
                    result = ohp.Content;
                    break;
                default:
                    result = new Item<Type>(x, indexes);
                    break;
            }
            Cache.Add(key, result);
            return result;
        }

        private Item(Tensor<Type> x, XList<Scalar<int>, int> indexes) : base("Item", x, indexes) {}

        public override void Backward(Scalar<Type> delta, Backpropagation bp) =>
            bp.PushGradientTo(x, Op.OneHot(x.Shape, Indexes, content: delta));

        public override Scalar<Type> Clone(IReadOnlyList<IExpr> inputs)
        {
            var _x = (Tensor<Type>)inputs[0];
            var _indexes = (XList<Scalar<int>, int>)inputs[1];
            return new Item<Type>(_x, _indexes);
        }
    }
}
