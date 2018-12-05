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

using Proxem.NumNet;
using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet.Operators.Tensors
{
    /// <summary>The representation of the slicing operations on Tensors.</summary>
    public class Slicing<Type> : Tensor<Type>.NAry
    {
        public static Tensor<Type> Create(Tensor<Type> x, IReadOnlyList<XSlice> slices)
        {
            if (slices.Count == 0) throw new RankException("no slices provided");

            if (x is BroadCast<Type> xBroad) // && slices.Count == 1 && slices[0].IsSingleton()
            {
                var broadcasted = xBroad.broadcast;
                var needBroadcast = new List<int>();
                var keptSlices = new List<XSlice>();

                for (int d = 0; d < slices.Count; ++d)
                {
                    if (broadcasted.Contains(d))
                    {
                        if (slices[d].IsSingleton)
                        {
                            keptSlices.Add(0);
                        }
                        else
                        {
                            keptSlices.Add(XSlicer._);
                            needBroadcast.Add(d);
                        }
                    }
                    else
                    {
                        keptSlices.Add(slices[d]);
                    }
                }

                if (needBroadcast.Count == 0) return xBroad.x[keptSlices];
                else return new Slicing<Type>(x, slices.ToArray());
            }

            var result = new Slicing<Type>(x, slices.ToArray());
            switch (x)
            {
                case Fill<Type> fill:
                    return Op.ConstLike(fill.x, result);
                default:
                    return result;
            }
        }

        private Slicing(Tensor<Type> x, XList<XSlice, Slice> slices)
            : base("[]", x, slices)
        {
            // count the number of dropped dimensions
            Shape = new Dim[x.NDim - slices.Count(s => s.IsSingleton)];

            // align new axis with old ones
            int i = 0, j = 0;
            foreach (var slice in slices)
            {
                if (!slice.IsSingleton)
                    Shape[i++] = slice.ExtractAxis(x, j);
                ++j;
            }
            // axis not appearing in slices are not modified
            for (; i < Shape.Length; ++i)
                Shape[i] = x.Shape[j++];
        }

        public override sealed Dim[] Shape { get; }

        public Tensor<Type> x => (Tensor<Type>)Inputs[0];

        public XList<XSlice, Slice> Slices => (XList<XSlice, Slice>)Inputs[1];

        public override void Backward(Tensor<Type> delta, Backpropagation bp) =>
            bp.PushGradientTo(x, Op.OneHot(x.Shape, Slices, delta));

        public override NAry Clone(IReadOnlyList<IExpr> inputs) =>
            new Slicing<Type>((Tensor<Type>)inputs[0], (XList<XSlice, Slice>)inputs[1]);
    }
}
