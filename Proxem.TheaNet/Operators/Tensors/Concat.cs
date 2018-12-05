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
using static Proxem.TheaNet.XSlicer;

namespace Proxem.TheaNet.Operators.Tensors
{
    /// <summary>
    /// Concat several tensors into one.
    /// </summary>
    public class Concat<T> : Tensor<T>.NAry
    {
        Tensor<T>[] _inputs;
        XSlice[] _slices;
        int _axis;

        public static Tensor<T> Create(int axis, Tensor<T>[] inputs)
        {
            if (inputs.Length == 1)
                return inputs[0];
            else
                return new Concat<T>(axis, inputs);
        }

        protected Concat(int axis, params Tensor<T>[] inputs) : base("Concat", (Scalar<int>)axis, inputs.ToStructArray())
        {
            int ndim = inputs.Max(i => i.NDim);
            if (axis < 0) axis += ndim;

            foreach (var x in inputs)
                x.AssertOfDim(ndim);

            this.Shape = new Dim[ndim];
            for (int a = 0; a < ndim; ++a)
                Shape[a] = inputs[0].Shape[a];
            _slices = new XSlice[inputs.Length];
            _slices[0] = Range(0, this.Shape[axis]);

            for (int i = 1; i  < inputs.Length; ++i)
            {
                inputs[i].AssertOfDim(ndim);
                for (int a = 0; a < ndim; ++a)
                    if (a != axis)
                        ShapeExtension.Bind(ref inputs[i].Shape[a], ref this.Shape[a]);

                var start = Shape[axis];
                Shape[axis] += inputs[i].Shape[axis];
                _slices[i] = Range(start, Shape[axis]);
            }

            _inputs = inputs;
            _axis = axis;
        }

        public override Dim[] Shape { get; }

        public Dim Axis => (Dim)Inputs[0];

        public override void Backward(Tensor<T> delta, Backpropagation bp)
        {
            delta.AssertOfShape(Shape);
            var slices = Enumerable.Repeat(_, NDim).ToArray();
            for (int i = 0; i < _inputs.Length; ++i)
            {
                slices[_axis] = _slices[i];
                // slices is copied before being passed to the indexer
                bp.PushGradientTo(_inputs[i], delta[slices.ToArray()]);
            }
        }

        public override NAry Clone(IReadOnlyList<IExpr> inputs) =>
            new Concat<T>(((Scalar<int>.Const)inputs[0]).Value, (XList<Tensor<T>, NumNet.Array<T>>)inputs[1]);
    }
}
