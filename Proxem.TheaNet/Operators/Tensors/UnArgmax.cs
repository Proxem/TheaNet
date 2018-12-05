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
    /// The reverse of the `Argmax` operator.
    /// Since the result of `Argmax` is a bit specialized, it requires a dedicated operator.
    /// </summary>
    public class UnArgmax<Type> : Tensor<Type>.NAry
    {
        int Axis;

        public static Tensor<T> Create<T>(Tensor<T> x, Tensor<int> indexes, int axis, Dim axisDim)
        {
            var dim = x.NDim;
            var shape = new Dim[dim];
            Array.Copy(x.Shape, shape, dim);
            shape[axis] = axisDim;
            return new UnArgmax<T>(x, indexes, axis, shape);
        }

        public static Tensor<T> Create<T>(Tensor<T> x, Tensor<int> indexes, int axis, Dim[] shape)
        {
            x.AssertOfDim(shape.Length);
            for (int i = 0; i < x.NDim; ++i)
                if (i != axis)
                    ShapeExtension.Bind(ref x.Shape[i], ref shape[i]);
            return new UnArgmax<T>(x, indexes, axis, shape);
        }

        private UnArgmax(Tensor<Type> x, Tensor<int> indexes, int axis, Dim[] shape)
            : base("UnArgmax", new IExpr[] { x, indexes, shape[axis] }, new object[] { axis.Named("axis") })
        {
            Axis = axis;
            Shape = shape;
        }

        public override sealed Dim[] Shape { get; }

        public Tensor<Type> x => (Tensor<Type>)this.Inputs[0];

        public Tensor<int> Indexes => (Tensor<int>)this.Inputs[1];

        public Dim AxisDim => (Dim)this.Inputs[2];

        public override void Backward(Tensor<Type> delta, Backpropagation bp)
        {
            throw new NotImplementedException();
        }

        public override NAry Clone(IReadOnlyList<IExpr> inputs) =>
            (NAry)Create((Tensor<Type>)inputs[0], (Tensor<int>)inputs[1], this.Axis, (Dim)inputs[2]);
    }
}

