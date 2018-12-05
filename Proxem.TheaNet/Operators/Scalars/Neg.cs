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
    /// <summary>Opposite of a scalar.</summary>    
    public class Neg<T> : Scalar<T>.Unary
    {
        /// <summary>
        /// Create a canonical representation for the opposite of a scalar.
        /// </summary>
        /// <remarks>
        /// - a => -a
        /// - -x => x
        /// </remarks>
        public static Scalar<T> Create(Scalar<T> x)
        {
            switch (x)
            {
                case Const constx:
                    return Numeric.Neg(constx.Value);
                case Neg<T> negx:
                    return negx.x;
                case Mul<T> mulx:
                    //    return (-mulx.x) * mulx.y;      // stack overflow
                    if (mulx.x is Sub<T> sub) return (sub.y - sub.x) * mulx.y;      // important for sigmoid
                    return new Neg<T>(x);
                default:
                    return new Neg<T>(x);
            }
        }

        public Neg(Scalar<T> x) : base("Neg", x, (_, f) => Numeric<T>.MinusOne) {}
    }
}
