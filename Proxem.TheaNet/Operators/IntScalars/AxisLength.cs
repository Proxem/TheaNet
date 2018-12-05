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

namespace Proxem.TheaNet.Operators.IntScalars
{
    public class AxisLength : Scalar<int>.NAry
    {
        internal AxisLength(ITensor x, int axis): base("Shape", x, (Scalar<int>)axis)
        {
            if (axis < -x.NDim || axis >= x.NDim)
                throw new IndexOutOfRangeException($"Axis {axis} is out of bound for {x}.Shape of length {x.NDim}");
            if (axis < 0) axis += x.NDim;
            this.Axis = axis;
        }

        public readonly int Axis;
        public ITensor x => (ITensor)Inputs[0];

        public override void Backward(Scalar<int> delta, Backpropagation bp) { }

        public override Scalar<int> Clone(IReadOnlyList<IExpr> inputs) => ((ITensor)inputs[0]).Shape[((Const)inputs[1]).Value];

    }
}
