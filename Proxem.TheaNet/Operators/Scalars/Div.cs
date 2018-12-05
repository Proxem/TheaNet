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
    /// <summary>Division between two scalars.</summary>
    public class Div<Type> : Scalar<Type>.Binary
    {
        /// <summary>
        /// Create a canonical representation for the division of two scalars.
        /// x / x => 1
        /// 0 / y => 0
        /// x / 1 => x
        /// x / float => (1 / float) * x
        /// (xx / xy) / y => xx / (xy * yy)
        /// x / (yx / yy) => (x * yy) / yx
        /// (x * y) / y => x
        /// </summary>
        public static Scalar<Type> Create(Scalar<Type> x, Scalar<Type> y)
        {
            bool isInt = typeof(Type) == typeof(int);

            // x / x => 1
            if (x == y) return Numeric<Type>.One;   // y == 0 ?
            // 0 / y => 0
            var consx = x as Const;
            if (consx != null && consx.Value.Equals(Numeric<Type>.Zero)) return consx;

            if (y is Const consy)
            {
                // x / 1 => x
                if (consy.Value.Equals(Numeric<Type>.One)) return x;
                if (consx != null)
                    return Numeric.Div(consx.Value, consy.Value);

                // x / float => (1 / float) * x
                if (!isInt)
                    return Numeric.Div(Numeric<Type>.One, consy.Value) * x;

                // now Type == int

                // a * xy / b if (a % b == 0) => (a / b) * xy
                if (x is Mul<Type> xMul && xMul.x is Const)
                {
                    int x_ = (xMul.x as Scalar<int>.Const).Value;
                    int y_ = (y as Scalar<int>.Const).Value;
                    if (x_ % y_ == 0)
                        return (x_ / y_) * (xMul.y as Scalar<int>) as Scalar<Type>;
                }
            }

            if (!isInt)
            {
                // (xx / xy) / y => xx / (xy * yy)
                if (x is Div<Type> xDiv) return xDiv.x / (xDiv.y * y);

                // x / (yx / yy) => (x * yy) / yx
                if (y is Div<Type> yDiv) return (x * yDiv.y) / yDiv.x;
            }

            {
                // (x * y) / y => x
                if (x is Mul<Type> xMul)
                {
                    if (xMul.x == y) return xMul.y;
                    if (xMul.y == y) return xMul.x;
                }

                // x / (x * y) => 1 / x
                if (y is Mul<Type> yMul)
                {
                    if (yMul.x == x) return Numeric<Type>.One / yMul.y;
                    if (yMul.y == x) return Numeric<Type>.One / yMul.x;
                }
            }
            return new Div<Type>(x, y);
        }

        public Div(Scalar<Type> x, Scalar<Type> y) : base(
            "Div", x, y,
            (_x, _y, _f) => Numeric<Type>.One / _y,
            (_x, _y, _f) => - _x / (_y * _y)
        ) {}
    }
}
