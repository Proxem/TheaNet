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
using Proxem.NumNet;
using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet.Operators.FloatTensors
{
    public class MaxPooling2d_new : Tuple2,
                                    ITuple<Tensor<float>, Array<float>, Tensor<int>, Array<int>>,
                                    ITensorTuple
    {
        public readonly int pool_h;
        public readonly int pool_w;
        public readonly bool ig;
        private Dim[] _shape1;
        private Dim[] _shape2;

        public MaxPooling2d_new(Tensor<float> x, int pool_h, int pool_w, bool ig) : base(x)
        {
            this.pool_h = pool_h;
            this.pool_w = pool_w;
            this.ig = ig;

            _shape1 = new Dim[] { x.Shape[0] / pool_h, x.Shape[1] / pool_w };
            _shape2 = new Dim[] { x.Shape[0] / pool_h, x.Shape[1] / pool_w, 2 };
            // TODO: case ignoreborder == false
            //if (ig == false)
            //{
            //    if (((x.Axes[0] ^ pool_h) >= 0) && (x.Axes[0] % pool_h != 0)) axis[0]++;
            //    if (((x.Axes[1] ^ pool_w) >= 0) && (x.Axes[1] % pool_w != 0)) axis[1]++;
            //}
        }

        public Tensor<float> x => (Tensor<float>)Inputs[0];

        public override void Process(IProcessor processor) =>
            processor.ProcessFunctionCall<Tuple<object>>(this, "NN.NewDownSample_MaxPooling2d", new object[] { pool_h, pool_w });

        public void Backward1(Tensor<float> delta, Backpropagation bp) =>
            bp.PushGradientTo(x, new Unpooling_new(delta, this.Item2(), pool_h, pool_w, ig));

        public void Backward2(Tensor<int> delta, Backpropagation bp)
        {
            throw new NotImplementedException();
        }

        public override Tuple2 Clone(IReadOnlyList<IExpr> inputs) => new MaxPooling2d_new((Tensor<float>)inputs[0], pool_h, pool_w, ig);

        Scalar<int>[] ITensorTuple.Shape(int item)
        {
            switch (item)
            {
                case 1: return _shape1;
                case 2: return _shape2;
                default: throw new IndexOutOfRangeException();
            }
        }

    }
}

