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

using Proxem.TheaNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proxem.TheaNet.Operators.Scalars
{
    /// <summary>Addition of two scalar.</summary>
    public class Add<Type> : Scalar<Type>.Binary
    {
        /// <summary>
        /// Create a canonical representation for the addition of two scalars.
        /// </summary>
        /// <remarks>
        /// 0 + y => y
        /// x + 0 => x
        /// x + x => 2 * x
        /// x + b => b + x
        /// a + b => c
        /// a + (b + y) => (a + b) + y
        /// -x +  y =>   y - x
        ///  x + -y =>   x - y
        /// -x + -y => -(x + y)
        /// </remarks>
        public static Scalar<Type> Create(Scalar<Type> x, Scalar<Type> y)
        {
            // 0 + y => y
            if (x.IsZero) return y;
            // x + 0 => x
            if (y.IsZero) return x;
            // x + x => 2 * x
            if (x == y) return Numeric<Type>.Two * x;

            if (y is Const cy)
                if (x is Const cx)
                    // a + b => c
                    return Numeric.Add(cx.Value, cy.Value);
                else
                    // x + b => b + x
                    return Create(y, x);

            // a + (b + y) => (a + b) + y
            if (y is Add<Type> add)
                return (x + add.x) + add.y;

            // -x +  y =>   y - x
            // (-x) + (-y) => (-y) - x => -(x + y)
            if (x is Neg<Type> negx)
                return y - negx.x;
            //  x + -y =>   x - y
            if (y is Neg<Type> negy)
                return x - negy.x;

            return new Add<Type>(x, y);
        }

        public Add(Scalar<Type> x, Scalar<Type> y) : base(
            "Add", x, y,
            dx: (_x, _y, _f) => Numeric<Type>.One,
            dy: (_x, _y, _f) => Numeric<Type>.One
        ) { }
    }
}
