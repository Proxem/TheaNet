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
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Proxem.NumNet;
using Proxem.TheaNet.Binding;

namespace Proxem.TheaNet.Operators.Tensors
{
    public class LogSumExp : Tensor<float>.Aggregate
    {
        /// <summary>
        /// Numerically stable shortcut of Log(Sum(Exp(a), axis, keepsDim).
        /// </summary>
        public LogSumExp(Tensor<float> x, int axis) : base("LogSumExp", x, axis) {}

        public override void Backward(Tensor<float> delta, Backpropagation bp) =>
            bp.PushGradientTo(x, delta * Op.Softmax(x, Axis));

        public override Aggregate Clone(Tensor<float> x, int axis)
        {
            return new LogSumExp(x, axis);
        }
    }
}
