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
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Proxem.NumNet;

namespace Proxem.TheaNet.Numerics
{
    public class Single : Numeric<float>
    {
        public override string GetLiteral(float a)
        {
            return string.Format(CultureInfo.InvariantCulture, "{0}f", a);
        }

        public override bool IsNegative(float a)
        {
            return a < 0;
        }

        public override float Neg(float a)
        {
            return -a;
        }

        public override float Add(float a, float b)
        {
            return a + b;
        }

        public override float Sub(float a, float b)
        {
            return a - b;
        }

        public override float Mul(float a, float b)
        {
            return a * b;
        }

        public override float Div(float a, float b)
        {
            return a / b;
        }

        public override bool IntegerDiv()
        {
            return false;
        }

        public override float GetScalar(string name)
        {
            return Runtime.Float[name];
        }

        public override void SetScalar(string name, float value)
        {
            Runtime.Float[name] = value;
        }

        public override Array<float> GetTensor(string name)
        {
            return Runtime.FloatArray[name];
        }

        public override void SetTensor(string name, Array<float> value)
        {
            Runtime.FloatArray[name] = value;
        }

        public override float Abs(float a)
        {
            return Math.Abs(a);
        }
    }
}
