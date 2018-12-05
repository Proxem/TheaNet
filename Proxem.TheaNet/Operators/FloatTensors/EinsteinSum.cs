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
using static Proxem.NumNet.EinsteinSumTools;

namespace Proxem.TheaNet.Operators.TTensors
{
    using FloatTensors;
    using Dim = Scalar<int>;

    public class EinsteinSum<T> : Tensor<T>.Binary<Tensor<T>, Array<T>, Tensor<T>, Array<T>>
    {
        public static Tensor<float> Create(Tensor<float> x, Tensor<float> y, string einstein)
        {
            var _einstein = EinsteinRead(einstein);

            return _simplifyDot(x, y, _einstein)
                ?? _simplifyOuter(x, y, _einstein)
                ?? new EinsteinSum<float>(x, y, einstein);
        }

        public readonly string einsteinString;
        Einstein[] einsteinRaw;
        Dim[] _shape;

        private EinsteinSum(Tensor<T> x, Tensor<T> y, string einstein): base("EinsteinSum", x, y, einstein)
        {
            einsteinString = einstein;
            var xyz = EinsteinSplit(einsteinString);
            x.AssertOfDim(xyz.Item1.Length);
            y.AssertOfDim(xyz.Item2.Length);

            einsteinRaw = EinsteinRead(einstein);
            _shape = EinsteinShape(x.Shape, y.Shape, einsteinRaw);
        }

        public override Dim[] Shape => _shape;

        public override Binary<Tensor<T>, Array<T>, Tensor<T>, Array<T>> Clone(Tensor<T> x, Tensor<T> y) =>
            new EinsteinSum<T>(x, y, einsteinString);

        public override void Backward(Tensor<T> delta, Backpropagation bp)
        {
            var xyz = EinsteinSplit(einsteinString);
            var zyx = $"{xyz.Item3},{xyz.Item2}->{xyz.Item1}";
            bp.PushGradientTo(x, new EinsteinSum<T>(delta, y, zyx));

            var xzy = $"{xyz.Item1},{xyz.Item3}->{xyz.Item2}";
            bp.PushGradientTo(y, new EinsteinSum<T>(x, delta, xzy));
        }

        static Dim[] EinsteinShape(Dim[] shapeX, Dim[] shapeY, Einstein[] einstein)
        {
            var ndim = einstein.Sum(e => e.axisZ == null ? 0 : 1);
            var shapeZ = new Dim[ndim];

            foreach (var e in einstein)
                switch (e.mode)
                {
                    case EinsteinMode.INNER:
                        ShapeExtension.Bind(ref shapeX[(int)e.axisX], ref shapeY[(int)e.axisY]);
                        if (e.axisZ != null) shapeZ[(int)e.axisZ] = 1;
                        break;
                    case EinsteinMode.ELEMENTWISE:
                        ShapeExtension.Bind(ref shapeX[(int)e.axisX], ref shapeY[(int)e.axisY]);
                        shapeZ[(int)e.axisZ] = shapeX[(int)e.axisX];
                        break;
                    case EinsteinMode.OUTERX:
                        shapeZ[(int)e.axisZ] = shapeX[(int)e.axisX];
                        break;
                    case EinsteinMode.OUTERY:
                        shapeZ[(int)e.axisZ] = shapeY[(int)e.axisY];
                        break;
                    case EinsteinMode.SUMX:
                    case EinsteinMode.SUMY:
                        if (e.axisZ != null) shapeZ[(int)e.axisZ] = 1;
                        break;
                }

            return shapeZ;
        }

        private static Tensor<float> _simplifyOuter(Tensor<float> x, Tensor<float> y, Einstein[] einstein)
        {
            var outerX = einstein.Where(e => e.mode == EinsteinMode.OUTERX).ToArray();
            var outerY = einstein.Where(e => e.mode == EinsteinMode.OUTERY).ToArray();

            if (einstein.Length == outerX.Length + outerY.Length)
            {
                if (outerX.All(e => e.axisZ == e.axisX) && outerY.All(e => e.axisZ == e.axisY + outerY.Length))
                    return Op.Outer(x, y);
                else if (outerY.All(e => e.axisZ == e.axisY) && outerX.All(e => e.axisZ == e.axisX + outerY.Length))
                    return Op.Outer(y, x);
            }

            return null;
        }

        private static Tensor<float> _simplifyDot(Tensor<float> x, Tensor<float> y, Einstein[] einstein)
        {
            // "ij,jk->ik" is a dot
            // "ji,jk->ik" is a dot
            int outerX = -1, inner = -1, outerY = -1;
            bool valid = true;
            for (int i = 0; i < einstein.Length; ++i)
            {
                switch (einstein[i].mode)
                {
                    case EinsteinMode.INNER:
                        if (inner < 0)
                            inner = i;
                        else
                            valid = false;
                        break;
                    case EinsteinMode.OUTERX:
                        if (outerX == -1)
                            outerX = i;
                        break;
                    case EinsteinMode.OUTERY:
                        outerY = i;
                        break;
                }
            }
            if (valid && inner >= 0)
            {
                if (x.NDim <= 2 && y.NDim <= 2)
                {
                    var transX = outerX >= 0 && einstein[outerX].axisX > einstein[inner].axisX;
                    var transY = outerY >= 0 && einstein[outerY].axisY < einstein[inner].axisY;
                    var transZ = outerX >= 0 && outerY >= 0 && einstein[outerX].axisZ > einstein[outerY].axisZ;
                    if(!transZ)
                        return Op.Dot(x, y, transposeX: transX, transposeY: transY);
                    else
                        return Op.Dot(y, x, transposeX: !transY, transposeY: !transX);
                }
                else
                    return Op.TensorDot(
                        x, new int[] { (int)einstein[inner].axisX },
                        y, new int[] { (int)einstein[inner].axisY }
                    );
            }
            return null;
        }
    }
}
