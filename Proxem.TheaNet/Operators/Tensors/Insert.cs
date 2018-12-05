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
using Proxem.TheaNet.Binding;
using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet.Operators.Tensors
{
    public class Insert<Type> : Tensor<Type>.Binary<Tensor<Type>, Array<Type>, Tensor<Type>, Array<Type>>
    {
        public int Axis;    // TODO: Scalar<int> Axis
        public int Index;   // TODO: Scalar<int> Index

        /// <summary>
        /// Inserts y into x at the given index along the given axis
        /// </summary>
        /// <param name="x"></param>
        /// <param name="index"></param>
        /// <param name="y"></param>
        /// <param name="axis"></param>
        public Insert(Tensor<Type> x, int index, Tensor<Type> y, int axis) : base("Insert", x, y, index, axis.Named("axis"))
        {
            if (x.NDim != y.NDim + 1) throw new RankException();
            this.Index = index;
            this.Axis = axis;

            Shape = new Scalar<int>[x.Shape.Length];
            Array.Copy(x.Shape, Shape, x.Shape.Length);
            Shape[axis] += 1;
        }

        public override sealed Dim[] Shape { get; }

        public override void Backward(Tensor<Type> delta, Backpropagation bp)
        {
            throw new NotImplementedException();
        }

        public override Binary<Tensor<Type>, Array<Type>, Tensor<Type>, Array<Type>> Clone(Tensor<Type> x, Tensor<Type> y) =>
            new Insert<Type>(x, Index, y, Axis);
    }
}
