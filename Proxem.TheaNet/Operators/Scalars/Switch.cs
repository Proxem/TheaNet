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

namespace Proxem.TheaNet.Operators.Scalars
{
    public static class Switch
    {
        public static Scalar<R> Create<T, R>(Scalar<T> mask, Scalar<R> ifTrue, Scalar<R> ifFalse)
        {
            return new Switch<T, R>(mask, ifTrue, ifFalse);
        }
    }

    /// <summary>Ternary operator based on the `mask` value.</summary>
    class Switch<T, R> : Scalar<R>.NAry
    {
        public Switch(Scalar<T> mask, Scalar<R> ifTrue, Scalar<R> ifFalse) : base("Switch", mask, ifTrue, ifFalse)
        {

        }

        public Scalar<int> mask => (Scalar<int>)Inputs[0];

        public Scalar<R> ifTrue => (Scalar<R>)Inputs[1];

        public Scalar<R> ifFalse => (Scalar<R>)Inputs[2];


        public override void Backward(Scalar<R> delta, Backpropagation bp)
        {
            bp.PushGradientTo(ifTrue,  Op.Switch(mask, delta, Numeric<R>.Zero));
            bp.PushGradientTo(ifFalse, Op.Switch(mask, Numeric<R>.Zero, delta));
        }

        public override Scalar<R> Clone(IReadOnlyList<IExpr> inputs)
        {
            return new Switch<T, R>((Scalar<T>)inputs[0], (Scalar<R>)inputs[1], (Scalar<R>)inputs[2]);
        }
    }
}
