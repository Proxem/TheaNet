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

namespace Proxem.TheaNet.Operators.Scalars
{
    /// <summary>Transforms a `Tensor` to a `Scalar`</summary>
    public class Aggregate<T> : Scalar<T>.NAry
    {
        Func<Tensor<T>, Scalar<T>, Tensor<T>> dx;

        public Aggregate(string name, Tensor<T> x,
            Func<Tensor<T>, Scalar<T>, Tensor<T>> dx,
            object[] extras = null) : base(name, new[] { x }, extras)
        {
            this.dx = dx;
        }

        public Tensor<T> x => (Tensor<T>)this.Inputs[0];

        public override Scalar<T> Clone(IReadOnlyList<IExpr> inputs) => new Aggregate<T>(this.FunctionName, (Tensor<T>)inputs[0], this.dx, this._extraInputs);

        public override void Backward(Scalar<T> delta, Backpropagation bp) =>
            bp.PushGradientTo(x, delta * Dx(x, this));

        Tensor<T> _Dx;
        public Tensor<T> Dx(Tensor<T> x, Scalar<T> f) => (_Dx = _Dx ?? dx(x, f));
    }
}
