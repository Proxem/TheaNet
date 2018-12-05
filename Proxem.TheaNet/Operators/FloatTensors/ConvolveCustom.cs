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
using Proxem.NumNet;

namespace Proxem.TheaNet.Operators.FloatTensors
{
    using Dim = Scalar<int>;
    using static Convolve;
    using Dims = XList<Scalar<int>, int>;

    public class ConvolveCustom : Tensor<float>.NAry
    {
        readonly Func<Tensor<float>, Tensor<float>, Tensor<float>> f, g, h;
        readonly Tensor<float> _z;
        readonly Var _x, _y;
        public readonly string description;

        /// <summary>
        /// Allows to create a custom convolution (ie replace the multiplication with any a custom op)
        /// </summary>
        /// <remarks>
        /// The generic case of convolution requiers Jacobian computation.
        /// As this not implemented in TheaNet, the user as to provide himself the "Jacobian".
        /// This operator make the following assumptions:
        ///  if:
        ///     z = ConvolveCustom(x, y, f)
        /// then:
        ///     dx = CorrelateCustom(dz, y, g)
        ///     dy = CorrelateCustom(dz, x, h)
        /// </remarks>
        public static Tensor<float> Create(
            Tensor<float> x,
            Tensor<float> y,
            Func<Tensor<float>, Tensor<float>, Tensor<float>> f,
            Func<Tensor<float>, Tensor<float>, Tensor<float>> g,
            Func<Tensor<float>, Tensor<float>, Tensor<float>> h,
            ConvMode mode = ConvMode.Full,
            string description = null
        ) =>
            Create(x, y, f, g, h, GetConvolveDim(x.Shape[0], y.Shape[0], mode), description);

        public static NAry Create(
            Tensor<float> x,
            Tensor<float> y,
            Func<Tensor<float>, Tensor<float>, Tensor<float>> f,
            Func<Tensor<float>, Tensor<float>, Tensor<float>> g,
            Func<Tensor<float>, Tensor<float>, Tensor<float>> h,
            Dim convDim,
            string description
        )
        {
            var _x = new Var(x.Shape.DropLeft(1), "_x");
            var _y = new Var(y.Shape.DropLeft(1), "_y");
            var _z = f(_x, _y);

            var _shape = convDim.Pad(_z.Shape);
            var lambda = new Lambda { Expr = _z, Vars = new[] { _x, _y } };

            return new ConvolveCustom(x, y, _x, _y, _z, f, g, h, _shape, lambda, description);
        }

        private ConvolveCustom(
            Tensor<float> x,
            Tensor<float> y,
            Var _x,
            Var _y,
            Tensor<float> _z,
            Func<Tensor<float>, Tensor<float>, Tensor<float>> f,
            Func<Tensor<float>, Tensor<float>, Tensor<float>> g,
            Func<Tensor<float>, Tensor<float>, Tensor<float>> h,
            Dims shape,
            Lambda lambda,
            string description
        ) : base("ConvolveCustom", new IExpr[] { x, y, shape }, extraInputs: new object[] { lambda })
        {
            this._x = _x; this._y = _y; this._z = _z;
            this.f = f; this.g = g; this.h = h;
            this.description = description;
        }

        public Tensor<float> x => (Tensor<float>)Inputs[0];
        public Tensor<float> y => (Tensor<float>)Inputs[1];
        public override Dim[] Shape => (Dims)Inputs[2];

        public override void Backward(Tensor<float> delta, Backpropagation bp)
        {
            // TODO handles case where this isn't a einstein conv.
            var xyz = EinsteinSumTools.EinsteinSplit(description);
            var zyx = $"{xyz.Item3},{xyz.Item2}->{xyz.Item1}";
            var zxy = $"{xyz.Item3},{xyz.Item1}->{xyz.Item2}";

            var dx = CorrelateCustom.Create(delta, y, g, x.Shape[0], description: zyx);
            bp.PushGradientTo(x, dx);

            var dy = CorrelateCustom.Create(delta, x, h, y.Shape[0], description: zxy);
            bp.PushGradientTo(y, dy);
        }

        public override NAry Clone(IReadOnlyList<IExpr> inputs) => Create(
            (Tensor<float>) inputs[0],
            (Tensor<float>) inputs[1],
            f, g, h,
            ((Dims) inputs[2]).Values[0],
            description
        );
    }

    class CorrelateCustom : Tensor<float>.NAry
    {
        readonly Func<Tensor<float>, Tensor<float>, Tensor<float>> f;
        readonly Tensor<float> _z;
        readonly Var _x, _y;
        public readonly string description;

        public static Tensor<float> Create(
            Tensor<float> x,
            Tensor<float> y,
            Func<Tensor<float>, Tensor<float>, Tensor<float>> f,
            Dim convDim,
            string description
        )
        {
            var _x = new Var(x.Shape.DropLeft(1), "_x");
            var _y = new Var(y.Shape.DropLeft(1), "_y");
            var _z = f(_x, _y);

            var _shape = convDim.Pad(_z.Shape);
            var lambda = new Lambda { Expr = _z, Vars = new[] { _x, _y } };
            return new CorrelateCustom(x, y, _x, _y, _z, f, _shape, lambda, description);
        }

        private CorrelateCustom(
            Tensor<float> x,
            Tensor<float> y,
            Var _x,
            Var _y,
            Tensor<float> _z,
            Func<Tensor<float>, Tensor<float>, Tensor<float>> f,
            Dims shape,
            Lambda lambda,
            string description
        ) : base("CorrelateCustom", new IExpr[] { x, y, shape }, new object[] { lambda })
        {
            this._x = _x; this._y = _y; this._z = _z;
            this.f = f;
            this.description = description;
        }

        public override Dim[] Shape => (Dims)Inputs[2];

        public override void Backward(Tensor<float> delta, Backpropagation bp)
        {
            throw new NotImplementedException();
        }

        public override NAry Clone(IReadOnlyList<IExpr> inputs) => (NAry)Create(
            (Tensor<float>)inputs[0],
            (Tensor<float>)inputs[1],
            f,
            ((Dims)inputs[2]).Values[0],
            description
        );
    }
}
