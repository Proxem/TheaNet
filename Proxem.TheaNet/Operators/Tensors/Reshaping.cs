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

namespace Proxem.TheaNet.Operators.Tensors
{
    using Dim = Scalar<int>;
    using Int = Scalar<int>.Const;

    /// <summary>Reshape a tensor.</summary>
    public class Reshaping<Type> : Tensor<Type>.Binary<Tensor<Type>, NumNet.Array<Type>, XList<Scalar<int>, int>, int[]>
    {
        public static Tensor<Type> Create(Tensor<Type> x, Dim[] shape)
        {
            int resizePos = -1;
            Dim size = 1;
            for (int i = 0; i < shape.Length; ++i)
            {
                if (shape[i].Check((Int c) => c.Value == -1))
                {
                    if (resizePos >= 0)
                        throw new ArgumentException("Can't reshape to: [" + string.Join(", ", shape.Select(a => a.ToString())) + "]");
                    resizePos = i;
                }
                else
                    size *= shape[i];
            }

            if (resizePos >= 0)
            {
                var originalSize = x.Shape.Aggregate((Dim)1, (s, d) => s * d);
                shape[resizePos] = originalSize / size;
            }

            if(x.NDim == shape.Length)
            {
                bool sameShape = true;
                for (int i = 0; i < shape.Length; ++i)
                    if (!ShapeExtension.WillEqualTo(x.Shape[i], shape[i]))
                        sameShape = false;
                if (sameShape)
                    return x;
            }

            switch (x)
            {
                case Fill<Type> fill:
                    return Op.Const(fill.x, shape);
                default:
                    return new Reshaping<Type>(x, shape);
            }
        }

        private Reshaping(Tensor<Type> x, XList<Scalar<int>, int> shape) : base("Reshape", x, shape)
        {
        }

        public override Dim[] Shape => y;

        public override void Backward(Tensor<Type> delta, Backpropagation bp) =>
            bp.PushGradientTo(x, delta.Reshape(x.Shape));

        public override Binary<Tensor<Type>, NumNet.Array<Type>, XList<Scalar<int>, int>, int[]> Clone(Tensor<Type> x, XList<Scalar<int>, int> y) =>
            new Reshaping<Type>(x, y);
    }
}
