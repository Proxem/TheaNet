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

namespace Proxem.TheaNet.Operators.Scalars
{
    /// <summary>Puts one scalar to the power of another.</summary>    
    public class Pow : Scalar<float>.Binary
    {
        public Pow(Scalar<float> x, Scalar<float> y) : base(
            "Pow", x, y,
            dx: (_x, _y, _f) => _y * Op.Pow(_x, _y - 1f),
            dy: (_x, _y, _f) => Op.Log(_x) * _f
        )
        { }
    }

    /// <summary>Compares two scalars (1 for true, 0 for false).</summary>    
    public class Gt<T> : Scalar<T>.Binary
    {
        public Gt(Scalar<T> x, Scalar<T> y) : base(
            "Gt", x, y,
            (_x, _y, _f) => Numeric<T>.Zero,
            (_x, _y, _f) => Numeric<T>.Zero
        ) { }
    }

    // TODO: merge with Gt ?
    // this two classes was necessary because Apply doesn't handle Binary and Unary the same way
    // is it still the case
    /// <summary>Compares a scalar to a constant (1 for true, 0 for false).</summary>    
    public class GtCst<T> : Scalar<T>.Unary
    {
        public GtCst(Scalar<T> x, Const y) : base(
            "Gt", x,
            (_x, _f) => Numeric<T>.Zero,
            extraInputs: new object[] { y.Value }
        ) { }

        public override void Backward(Scalar<T> delta, Backpropagation bp) { }
    }

    /// <summary>Compares two scalars with $\gteq$ (1 for true, 0 for false).</summary>    
    public class GtEq<T> : Scalar<T>.Binary
    {
        public GtEq(Scalar<T> x, Scalar<T> y) : base(
            "Ge", x, y,
            (_x, _y, _f) => Numeric<T>.Zero,
            (_x, _y, _f) => Numeric<T>.Zero
        ) { }
    }
}
