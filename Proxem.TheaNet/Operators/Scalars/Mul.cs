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
    /// <summary>Multiplication between two scalars.</summary>    
    public class Mul<Type> : Scalar<Type>.Binary
    {
        /// <summary>
        /// Create a canonical representation for the multiplication of two scalars.
        /// </summary>
        /// <remarks>
        /// 0 * y => 0
        /// 1 * y => y
        /// -1 * y => -y
        /// a * ( b * yy) => (a * b) * yy
        /// a * ( - y) => (- a ) * y
        /// x * b => b * x
        /// a * ( b / yy) => (a * b) / yy
        /// x.Item[z] * y.Item[z] => (x * y).Item[z]
        /// </remarks>
        public static Scalar<Type> Create(Scalar<Type> x, Scalar<Type> y)
        {
            if (x == y) return Op.Square(x);
            var consx = x as Const;
            if (consx != null)
            {
                // 0 * y => 0
                if (consx.IsZero) return consx;
                // 1 * y => y
                if (consx.IsOne) return y;

                // a * (b * yy) => (a * b) * yy
                var yMul = y as Mul<Type>;
                if (yMul != null && yMul.x is Const) return (x * yMul.x) * yMul.y;

                // a * (b / yy) => (a * b) / yy;
                var yDiv2 = y as Div<Type>;
                if (yDiv2 != null && yDiv2.x is Const) return (x * yDiv2.x) / yDiv2.y; // important for softmax gradient optimization

                // -1 * y => -y
                if (consx.IsMinusOne) return -y;

                // a * ( - y) => (- a ) * y
                var yNeg = y as Neg<Type>;
                if (yNeg != null) return Create(-x, yNeg.x);
            }

            if (y is Const consy)
            {
                if (consx != null) return Numeric.Mul(consx.Value, consy.Value);
                // x * b => b * x
                else return Create(y, x);
            }

            // a * ( b / yy) => (a * b) / yy
            if (y is Div<Type> yDiv && !(y is Div<int>)) return (x * yDiv.x) / yDiv.y;

            if (x is Div<Type> xdiv)
            {
                // (a / y) * y => a
                if (xdiv.y == y) return xdiv.x; // important for softmax gradient optimization
                var ymul = y as Mul<Type>;
                if (ymul != null)
                {
                    // (xdiv.x / xdiv.y) * (ymul.x * ymul.y) => (xdiv.x * ymul.x * ymul.y) / xdiv.y
                    if (ymul.y == xdiv.y) return xdiv.x * ymul.x;   // important for sigmoid gradient optimization
                    if (ymul.x == xdiv.y) return xdiv.x * ymul.y;
                    return (xdiv.x * ymul.x * ymul.y) / xdiv.y;
                }
                return (xdiv.x * y) / xdiv.y;
            }

            // (-x) * y => -(x * y)
            if (x is Neg<Type> xneg) return Neg<Type>.Create(xneg.x * y);

            // x.Item[z] * y.Item[z] => (x * y).Item[z] // important for softmax gradient optimization
            if (x is Item<Type> xitem)
            {
                var yitem = y as Item<Type>;
                if (yitem != null && xitem.Indexes.SequenceEqual(yitem.Indexes))
                {
                    return (xitem.x * yitem.x).Item[xitem.Indexes];
                }
            }

            // (a / b) * (b * c) => (a * c)
            if (x is Div<Type> xDiv)
            {
                var yMul = y as Mul<Type>;
                if (yMul != null && xDiv.y == yMul.x) return xDiv.x * yMul.y;
            }

            return new Mul<Type>(x, y);
        }

        public Mul(Scalar<Type> x, Scalar<Type> y) : base(
            "Mul", x, y,
            (_x, _y, _f) => _y,
            (_x, _y, _f) => _x
        ) { }
    }
}
