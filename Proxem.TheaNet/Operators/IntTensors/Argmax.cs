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
using System.Threading.Tasks;
using Proxem.NumNet;
using Proxem.TheaNet.Binding;

using Dim = Proxem.TheaNet.Scalar<int>;
using Proxem.TheaNet.Operators.Tensors;


namespace Proxem.TheaNet.Operators.IntTensors
{
    /// <summary>
    /// Aggregate Argmax along an axis.
    /// see https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
    /// </summary>
    public class Argmax<U_> : Tensor<int>.Unary<Tensor<U_>, Array<U_>>
    {
        public readonly int Axis;
        Dim[] _shape;

        public static Tensor<int> Create(Tensor<U_> x, int axis)
        {
            if (axis < 0) axis += x.NDim;
            return new Argmax<U_>(x, axis);
        }

        private Argmax(Tensor<U_> x, int axis) : base("Argmax", x, new object[] { axis.Named("axis") })
        {
            Axis = axis;
            _shape = x.Shape.ToArray();
            _shape[axis] = 1;
        }

        public override Dim[] Shape => this._shape;

        public override void Backward(Tensor<int> delta, Backpropagation bp) {}

        public override Unary<Tensor<U_>, Array<U_>> Clone(Tensor<U_> x)
        {
            return new Argmax<U_>(x, Axis);
        }
    }
}
