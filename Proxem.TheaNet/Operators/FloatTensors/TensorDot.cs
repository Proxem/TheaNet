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

namespace Proxem.TheaNet.Operators.FloatTensors
{
    using Tensors;
    using Axes = XList<Scalar<int>, int>;
    using Dim = Scalar<int>;

    public class TensorDot : Tensor<float>.NAry
    {
        Axes axesX, axesY;
        // TODO use (to avoid runtime shape checking reshape and transpose)
        //int n;
        Tensor<float> xt, yt;
        int[] keptX, keptY;
        private readonly Dim[] _shape;

        public static Tensor<float> Create(Tensor<float> a, IEnumerable<int> axesA, Tensor<float> b, IEnumerable<int> axesB)
        {
            var removeA = axesA.Select(i => i < 0 ? i + a.NDim : i).ToArray();
            var removeB = axesB.Select(i => i < 0 ? i + b.NDim : i).ToArray();

            int n = removeA.Length;

            if (removeB.Length != n)
                throw new RankException(string.Format(
                    "The axes parameters of TensorDot should have the same size. Found [{0}] and [{1}].",
                    string.Join(", ", removeA.AsEnumerable()), string.Join(", ", removeB.AsEnumerable())));

            for (int d = 0; d < n; ++d)
                ShapeExtension.Bind(ref a.Shape[removeA[d]], ref b.Shape[removeB[d]]);

            var keptX = Enumerable.Range(0, a.NDim).Where(d => !removeA.Contains(d)).ToArray();
            var keptY = Enumerable.Range(0, b.NDim).Where(d => !removeB.Contains(d)).ToArray();
            // Move the axes to sum over to the end of "a"
            var keptA = Enumerable.Range(0, a.NDim).Where(d => !removeA.Contains(d));
            var at = a.DimShuffle(keptA.Concat(removeA).ToArray());

            // Move the axes to sum over to the front of "b"
            var keptB = Enumerable.Range(0, b.NDim).Where(d => !removeB.Contains(d));
            var bt = b.DimShuffle(removeB.Concat(keptB).ToArray());

            var resultShape = keptA.Select(axis => a.Shape[axis]).Concat(keptB.Select(axis => b.Shape[axis])).ToArray();
            var a2d = Reshape2D(at, a.NDim - n);
            var b2d = Reshape2D(bt, n);
            var res = Op.Dot(a2d, b2d).Reshape(resultShape);
            res.Comment = "TensorDot";

            return res;
            //return new TensorDot(a, ToAxes(_axesX), b, ToAxes(_axesY));
        }

        private static Tensor<T> Reshape2D<T>(Tensor<T> a, int dimensionsAsRows)
        {
            if (dimensionsAsRows < 0) dimensionsAsRows += a.NDim;
            var before = a.Shape.Take(dimensionsAsRows).Aggregate((Dim)1, (x, y) => x * y);
            var after = a.Shape.Skip(dimensionsAsRows).Aggregate((Dim)1, (x, y) => x * y);
            return a.Reshape(new[] { before, after });
        }

        private TensorDot(Tensor<float> x, Axes axesX, Tensor<float> y, Axes axesY) : base("TensorDot", x, axesX, y, axesY)
        {
            this.axesX = axesX;
            this.axesY = axesY;

            var _axesX = ToInt(axesX);
            var _axesY = ToInt(axesY);
            keptX = Enumerable.Range(0, x.NDim).Where(d => !_axesX.Contains(d)).ToArray();
            keptY = Enumerable.Range(0, y.NDim).Where(d => !_axesY.Contains(d)).ToArray();
            _shape = keptX.Select(axis => x.Shape[axis]).Concat(keptY.Select(axis => y.Shape[axis])).ToArray();
        }

        private static int[] ToInt(Axes axes) => axes.Select(a => ((Scalar<int>.Const)a).Value).ToArray();
        private static Axes ToAxes(int[] axes) => new Axes(axes.Select(a => Op.Const(a)).ToArray());

        public override Dim[] Shape => this._shape;
        public Tensor<float> x => (Tensor<float>)Inputs[0];
        public Tensor<float> y => (Tensor<float>)Inputs[2];

        public override void Backward(Tensor<float> delta, Backpropagation bp)
        {
            delta.AssertOfShape(Shape);

            // First we Shuffle x and y in a TensorDot friendly manner.
            // This shuffling doesn't change the result of the TensorDot but make the gradient easier to compute.
            xt = xt ?? x.DimShuffle(keptX.Union(ToInt(axesX)).ToArray());
            yt = yt ?? y.DimShuffle(ToInt(axesY).Union(keptY).ToArray());

            var deltaXt = Op.TensorDot(delta, Enumerable.Range(keptX.Length, keptY.Length), y, keptY);
            bp.PushGradientTo(xt, deltaXt);

            var deltaYt = Op.TensorDot(x, keptX, delta, Enumerable.Range(0, keptX.Length));
            bp.PushGradientTo(yt, deltaYt);
        }

        public override NAry Clone(IReadOnlyList<IExpr> inputs) => new TensorDot(
            (Tensor<float>)inputs[0],
            (Axes)inputs[1],
            (Tensor<float>)inputs[2],
            (Axes)inputs[3]
        );
    }
}
