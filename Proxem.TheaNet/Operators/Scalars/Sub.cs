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
    /// <summary>Substraction between two scalars.</summary>    
    public class Sub<Type> : Scalar<Type>.Binary
    {
        private static Dictionary<string, Scalar<Type>> Cache = new Dictionary<string, Scalar<Type>>();

        /// <summary>
        /// Create a canonical representation for the substraction of two scalars.
        /// </summary>
        /// <remarks>
        /// x - x => 0
        /// x - 0 => x
        /// 0 - y => -y
        /// a - b => c
        /// x - b => -b + x
        /// x - -y => x + y
        /// -x - y => -(x + y)
        /// </remarks>
        public static Scalar<Type> Create(Scalar<Type> x, Scalar<Type> y)
        {
            string key = $"Sub|{x.Id}|{y.Id}";
            Scalar<Type> result;
            if (Cache.TryGetValue(key, out result)) return result;
            result = _Create(x, y);
            Cache.Add(key, result);
            return result;
        }

        // TODO: make inner function
        private static Scalar<Type> _Create(Scalar<Type> x, Scalar<Type> y)
        {
            // x - x => 0
            if (x == y) return Numeric<Type>.Zero;
            // x - 0 => x
            if (y.IsZero) return x;
            // 0 - y => -y
            if (x.IsZero) return -y;

            if (y is Const cy)
                // a - b => c
                if (x is Const cx)
                    return Numeric<Type>.Current.Sub(cx.Value, cy.Value);
                // x - b => -b + x
                else
                    return (-y) + x;

            // x - -y => x + y
            if (y is Neg<Type>) return x + (-y);
            // -x - y => -(x + y)
            if (x is Neg<Type>) return -(-x + y);

            return new Sub<Type>(x, y);
        }

        public Sub(Scalar<Type> x, Scalar<Type> y) : base(
            "Sub", x, y,
            (_x, _y, _f) => Numeric<Type>.One,
            (_x, _y, _f) => Numeric<Type>.MinusOne
        ) { }
    }
}
