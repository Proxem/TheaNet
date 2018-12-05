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

namespace Proxem.TheaNet.Operators.Scalars
{
    /// <summary>
    /// Converts a `Scalar{T}` to `Scalar{U}`.
    /// </summary>
    public class Cast<T, U> : Scalar<T>.NAry
    {
        public static Scalar<T> Create(Scalar<U> x)
        {
            if (x is Scalar<U>.Const @const) return (T)Convert.ChangeType(@const.Value, typeof(T));
            return new Cast<T, U>(x);
        }

        private Cast(Scalar<U> x) : base("CastScalar", x)
        {
        }

        public Scalar<U> x => (Scalar<U>)this.Inputs.First();

        public override void Backward(Scalar<T> delta, Backpropagation bp) =>
            bp.PushGradientTo(x, delta.As<U>());

        public override Scalar<T> Clone(IReadOnlyList<IExpr> inputs) => new Cast<T, U>((Scalar<U>)inputs[0]);
    }
}
