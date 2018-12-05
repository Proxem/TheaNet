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
using Proxem.NumNet;

namespace Proxem.TheaNet.Numerics
{
    public class Int32 : Numeric<int>
    {
        public override string GetLiteral(int a)
        {
            return a.ToString();
        }

        public override bool IsNegative(int a)
        {
            return a < 0;
        }
        public override int Neg(int a)
        {
            return -a;
        }

        public override int Add(int a, int b)
        {
            return a + b;
        }

        public override int Sub(int a, int b)
        {
            return a - b;
        }

        public override int Mul(int a, int b)
        {
            return a * b;
        }

        public override int Div(int a, int b)
        {
            return a / b;
        }

        public override bool IntegerDiv()
        {
            return true;
        }

        public override int GetScalar(string name)
        {
            return Runtime.Int[name];
        }

        public override void SetScalar(string name, int value)
        {
            Runtime.Int[name] = value;
        }

        public override Array<int> GetTensor(string name)
        {
            return Runtime.IntArray[name];
        }

        public override void SetTensor(string name, Array<int> value)
        {
            Runtime.IntArray[name] = value;
        }

        public override int Abs(int a)
        {
            return Math.Abs(a);
        }
    }
}
