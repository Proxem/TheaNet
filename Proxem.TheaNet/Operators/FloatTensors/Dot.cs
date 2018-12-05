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
using Proxem.TheaNet.Operators.Tensors;
using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet.Operators.FloatTensors
{
    public class Dot : Tensor<float>.Binary<Tensor<float>, Array<float>, Tensor<float>, Array<float>>
    {
        public bool TransposeX;
        public bool TransposeY;
        private readonly Dim[] _shape;

        public static Tensor<float> Create(Tensor<float> x, Tensor<float> y, bool transposeX = false, bool transposeY = false)
        {
            if (!transposeX)
            {
                if (y is Fill<float> filly && filly.NDim == 0)   // TODO: more general optimization
                {
                    return filly.x * x;     // TODO: check shapes;
                }
            }
            var oneHotX = transposeX ? null : x as OneHot<float>;
            if (oneHotX != null)
                return Op.OneHot(new Dot(x, y, false, transposeY).Shape, oneHotX.Index, Create(oneHotX.Content, y, false, transposeY));

            if (x.IsZero || y.IsZero)
                return Op.ZerosLike(new Dot(x, y, transposeX, transposeY));

            ////else if (oneHotY != null)
            ////{
            ////    return new OneHot<float>(new Dot(x, y, TransposeX, false), oneHotY.Index, new Dot(x, oneHotY.a, TransposeX, tr));
            ////}
            //else
            if (x.NDim == 1 && y.NDim > 1)
                return new Dot(y, x, transposeX: !transposeY, transposeY: false);
            else if(x.NDim > 2 || y.NDim > 2)
            {
                if((transposeX && x.NDim > 2) || (transposeY && y.NDim > 2)) throw new NotImplementedException();

                if(transposeX && x.NDim == 1) x = x.Reshape(1, -1);
                if(transposeY && y.NDim == 1) y = y.Reshape(1, -1);

                var axesX = new int[]{-1};
                var axesY = new int[]{0};
                if(transposeX && x.NDim == 2) axesX[0] = 0;
                if(transposeY && y.NDim == 2) axesY[0] = 1;

                return Op.TensorDot(x, axesX, y, axesY);
            }
            else
                return new Dot(x, y, transposeX, transposeY);
        }

        private Dot(Tensor<float> x, Tensor<float> y, bool transposeX = false, bool transposeY = false)
            : base("Dot", x, y, transposeX.Named("transA"), transposeY.Named("transB"))
        {
            this.TransposeX = transposeX;
            this.TransposeY = transposeY;

            if (x.NDim == 1 && y.NDim == 1)
            {
                // rowV dot rowV (forbidden)
                if (TransposeX && TransposeY)
                    throw Rank(this, "Can't dot two row vectors: {0} and {1}");
                // colV dot colV (used as inner product)
                if (!TransposeX && !TransposeY)
                    TransposeX = true;
                // colV dot rowV (outer product)
                if (!TransposeX && TransposeY)
                    _shape = new Dim[] { x.Shape[0], y.Shape[0] };
                // rowV dot colV (inner product)
                if (TransposeX && !TransposeY)
                {
                    if (!x.Shape[0].CanEqualTo(y.Shape[0]))
                        throw Rank(this, "Can't dot {0} with {1}");
                    ShapeExtension.Bind(ref x.Shape[0], ref y.Shape[0]);
                    _shape = new Dim[] { };
                }
                return;
            }

            // mat dot colV
            if (y.NDim == 1)
            {
                TransposeY = false;
                if (!x.Shape[TransposeX ? 0 : x.NDim - 1].CanEqualTo(y.Shape[0]))
                    throw Rank(this, "Can't dot {0} with {1}");
                ShapeExtension.Bind(ref x.Shape[TransposeX ? 0 : x.NDim - 1], ref y.Shape[0]);
                _shape = (TransposeX ? x.Shape.Reverse().ToArray() : x.Shape).DropRight(1);
                return;
            }
            else if (y.NDim == 0)
            {
                TransposeY = false;
                _shape = x.Shape.ToArray();
            }
            // mat dot mat
            else
            {
                if (x.NDim == 1)
                    TransposeX = true;
                _shape = new Dim[x.NDim + y.NDim - 2];
                var axisX = x.Shape[TransposeX ? 0 : x.NDim - 1];
                var axisY = y.Shape[TransposeY ? 1 : y.NDim - 2];
                if (!axisX.CanEqualTo(axisY))
                    throw Rank(this, "Can't dot {0} with {1}");
                ShapeExtension.Bind(ref axisX, ref axisY);

                // copy x dims but (n - 1)
                if (!TransposeX)
                    for (int i = 0; i < x.NDim - 1; ++i)
                        _shape[i] = x.Shape[i];
                else
                    for (int i = 0; i < x.NDim - 1; ++i)
                        _shape[i] = x.Shape[x.NDim - 1 - i];

                // copy y dims but (n - 2)
                if (!TransposeY)
                {
                    for (int i = 0; i < y.NDim - 2; ++i)
                        _shape[x.NDim + i] = y.Shape[i];
                    _shape[x.NDim + y.NDim - 3] = y.Shape[y.NDim - 1];
                }
                else
                {
                    for (int i = 0; i < y.NDim - 2; ++i)
                        _shape[x.NDim + i] = y.Shape[y.NDim - 1 - i];
                    _shape[x.NDim + y.NDim - 3] = y.Shape[0];
                }
            }
        }

        public override Dim[] Shape => this._shape;

        public RankException Rank<T>(IExpr<T> target, string format)
        {
            return new RankException(string.Format(format,
                x.Shape.Format(target) + (TransposeX ? ".T" : ""),
                y.Shape.Format(target) + (TransposeY ? ".T" : "")
            ));
        }

        public override void Backward(Tensor<float> delta, Backpropagation bp)
        {
            Tensor<float> deltaX, deltaY;
            if (x.NDim > 2 || y.NDim > 2) throw new NotImplementedException("Backward of tensor dot");
            if (!TransposeX)
                deltaX = Create(delta, y, false, !TransposeY);
            else
                deltaX = Create(y, delta, TransposeY, true);
            bp.PushGradientTo(x, deltaX);

            if (!TransposeY)
                deltaY = Create(x, delta, !TransposeX, false);
            else
                deltaY = Create(delta, x, true, TransposeX);

            bp.PushGradientTo(y, deltaY);
        }

        public override Binary<Tensor<float>, Array<float>, Tensor<float>, Array<float>> Clone(Tensor<float> x, Tensor<float> y) =>
            new Dot(x, y, TransposeX, TransposeY);
    }
}
