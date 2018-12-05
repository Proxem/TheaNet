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

using Proxem.NumNet;
using Dim = Proxem.TheaNet.Scalar<int>;
using Proxem.TheaNet.Binding;

namespace Proxem.TheaNet.Operators.IntTensors
{
    /// <summary> Creates a range of integers. </summary>
    public class Range : Tensor<int>.Binary<Scalar<int>, int, Scalar<int>, int>
    {
        public Range(Scalar<int> start, Scalar<int> stop) : base("Range", start, stop)
        {
            this.Shape = new Dim[] { stop - start };
        }

        public override Dim[] Shape { get; }

        public override void Backward(Tensor<int> delta, Backpropagation bp)
        {
            throw new NotImplementedException();
        }

        public override Binary<Dim, int, Dim, int> Clone(Dim x, Dim y) =>
            new Range(x, y);
    }
}
