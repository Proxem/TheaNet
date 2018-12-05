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
using Proxem.TheaNet.Operators.FloatTensors;
using Proxem.TheaNet.Operators.IntTensors;
using Proxem.TheaNet.Operators.Tensors;
using Proxem.TheaNet.Operators.TTensors;
using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet
{
    public partial class Op
    {
        public static Tensor<T>.Var Vector<T>(string name)
        {
            CheckName(name);
            return new Tensor<T>.Var(1, name);
        }

        public static Tensor<T>.Var Vector<T>(Dim size, string name)
        {
            CheckName(name);
            return new Tensor<T>.Var(new Dim[] { size }, name);
        }

        public static Tensor<T>.Var Matrix<T>(string name)
        {
            CheckName(name);
            return new Tensor<T>.Var(2, name);
        }

        public static Tensor<T>.Var Matrix<T>(Dim rows, Dim cols, string name)
        {
            CheckName(name);
            return new Tensor<T>.Var(new Dim[] { rows, cols }, name);
        }

        public static Tensor<T>.Var Tensor3<T>(string name)
        {
            CheckName(name);
            return new Tensor<T>.Var(3, name);
        }

        public static Tensor<T>.Var Tensor3<T>(Dim stacks, Dim rows, Dim cols, string name)
        {
            CheckName(name);
            return new Tensor<T>.Var(new Dim[] { stacks, rows, cols}, name);
        }

        public static Tensor<T>.Shared Shared<T>(Array<T> value, string name)
        {
            CheckName(name);
            return new Tensor<T>.Shared(value, name);
        }

        public static Tensor<T> Abs<T>(Tensor<T> x) => Apply(x, _x => Abs(_x));

        public static Tensor<float> Convolve(Tensor<float> x, Tensor<float> y, ConvMode mode = ConvMode.Full)
        {
            return new Convolve(x, y, mode);
        }

        public static Tensor<float> Correlate(Tensor<float> x, Tensor<float> y, ConvMode mode = ConvMode.Valid)
        {
            return new Correlate(x, y, mode);
        }

        public static Tensor<float> Convolve2d(Tensor<float> x, Tensor<float> y, ConvMode mode = ConvMode.Full)
        {
            return new Convolve2d(x, y, mode);
        }

        public static Tensor<float> Correlate2d(Tensor<float> x, Tensor<float> y, ConvMode mode = ConvMode.Valid)
        {
            return new Correlate(x, y, mode);
        }

        public static Tensor<float> MaxPooling2d(Tensor<float> x, int pool_h, int pool_w, bool ig)
        {
            return new MaxPooling2d(x, pool_h, pool_w, ig);
        }

        public static ITuple<Tensor<float>, Array<float>,Tensor<int>, Array<int>> MaxPooling2d_new(Tensor<float> x, int pool_h, int pool_w, bool ig)
        {
            return new MaxPooling2d_new(x, pool_h, pool_w, ig);
        }

        public static Tensor<float> Unpooling_new(Tensor<float> x, Tensor<int> switches, int pool_h, int pool_w, bool ig)
        {
            return new Unpooling_new(x, switches, pool_h, pool_w, ig);
        }

        public static Tensor<float> Unpooling(Tensor<float> x, Tensor<float> y, int pool_h, int pool_w, bool ig)
        {
            return new Unpooling(x, y, pool_h, pool_w, ig);
        }

        public static Tensor<float> DimShuffle(Tensor<float> x, params int[] axes) => x.DimShuffle(axes);

        public static Tensor<T> ShapePadLeft<T>(Tensor<T> x, int pad)
        {
            if (pad == 0)
                return x;
            else
                return x.DimShuffle(Enumerable.Repeat((int)'x', pad).Concat(Enumerable.Range(0, x.NDim)).ToArray());
        }

        public static Tensor<float> Dot(Tensor<float> x, Tensor<float> y, bool transposeX = false, bool transposeY = false)
        {
            return Operators.FloatTensors.Dot.Create(x, y, transposeX, transposeY);
        }

        public static Tensor<float> TensorDot(Tensor<float> x, IEnumerable<int> axesX, Tensor<float> y, IEnumerable<int> axesY) =>
            Operators.FloatTensors.TensorDot.Create(x, axesX, y, axesY);

        public static Tensor<float> Exp(Tensor<float> x)
        {
            return Apply(x, Op.Exp);
        }

        public static Tensor<float> Log(Tensor<float> x)
        {
            return Apply(x, Op.Log);
        }

        public static Tensor<float> Clip(Tensor<float> x, float min, float max)
        {
            return Apply(x, _x => Op.Clip(_x, min, max));
        }

        public static Tensor<T> Eq<T>(Tensor<T> x, Tensor<T> y)
        {
            return (x >= y) * (x <= y);
        }

        public static Tensor<T> Neq<T>(Tensor<T> x, Tensor<T> y)
        {
            return Numeric<T>.One - Eq(x, y);
        }

        public static Tensor<float> Max(Tensor<float> x, int axis, bool keepDims = false)
        {
            var res = new Max<float>(x, axis);
            if (!keepDims)
                return res.Reshape(res.Shape.DropAt(axis));
            else
                return res;
        }

        public static Tensor<T> Const<T>(T content, params Dim[] shape) => Const((Scalar<T>)content, shape);

        public static Tensor<T> Const<T>(Scalar<T> content, params Dim[] shape) =>
            Fill<T>.Create(content, shape);

        public static Tensor<T> ConstLike<T>(Scalar<T> content, Tensor<T> shape) => Const(content, shape.Shape);

        public static Tensor<T> Zeros<T>(params Dim[] shape) => Const(Numeric<T>.Zero, shape);

        public static Tensor<T> ZerosLike<T>(Tensor<T> x) => Zeros<T>(x.Shape);

        public static Tensor<T> Ones<T>(params Dim[] shape) => Const(Numeric<T>.One, shape);

        public static Tensor<T> OnesLike<T>(Tensor<T> x) => Ones<T>(x.Shape);

        /// <summary>Creates a tensor where only one index of the first axis is not zero</summary>
        public static Tensor<T> OneHot<T>(Dim[] shape, Scalar<int> index, Tensor<T> content) =>
            Operators.Tensors.OneHot<T>.Create(shape, index, content);

        /// <summary>Creates a tensor where only some slices is not zero</summary>
        public static Tensor<T> OneHot<T>(Dim[] shape, XList<XSlice, Slice> slices, Tensor<T> content) =>
            OneHotSlice<T>.Create(shape, slices, content);

        /// <summary>Creates a tensor where only one point is not zero (indexes.Length == shape.Length)</summary>
        public static Tensor<T> OneHot<T>(Dim[] shape, Scalar<int>[] indexes, Scalar<T> content) =>
            OneHotPoint<T>.Create(shape, indexes, content);

        public static Tensor<float> Pow(Tensor<float> x, float a)
        {
            if (a == 1) return x;
            if (a == 2) return Op.Square(x);
            return Apply(x, ConstLike(a, x), (_x, _y) => new Operators.Scalars.Pow(_x, _y));
        }

        // TODO: move to NNet
        public static Tensor<float> Sigmoid(Tensor<float> x)
        {
            return Apply(x, Op.Sigmoid);
        }

        public static Tensor<float> Softmax(Tensor<float> x, int axis = -1)
        {
            return Operators.FloatTensors.Softmax.Create(x, axis);
        }

        public static Tensor<float> LogSumExp(Tensor<float> x, int axis = -1, bool keepDims = false)
        {
            var res = new LogSumExp(x, axis);
            if (!keepDims)
                return res.Reshape(res.Shape.DropAt(axis));
            else
                return res;
        }

        public static Tensor<float> Sqrt(Tensor<float> x) => Apply(x, _x => Sqrt(_x));
        public static Tensor<Type> Square<Type>(Tensor<Type> x) => Apply(x, _x => Square(_x));

        public static Tensor<Type> Sum<Type>(Tensor<Type> x, int axis, bool keepDims = false)
        {
            var res = Operators.Tensors.Sum<Type>.Create(x, axis);
            if (!keepDims)
                return res.Reshape(res.Shape.DropAt(axis));
            else
                return res;
        }

        public static Tensor<Type> Mean<Type>(Tensor<Type> x, int axis, bool keepDims = false) => Sum(x, axis, keepDims) / x.Shape[axis].As<Type>();

        public static Tensor<float> Tanh(Tensor<float> x) => Apply(x, Tanh);

        public static Tensor<int> Argmax<T>(Tensor<T> x, int axis, bool keepDims = false)
        {
            var arg = Operators.IntTensors.Argmax<T>.Create(x, axis);
            if (!keepDims)
                arg = arg.Reshape(arg.Shape.DropAt(axis));
            return arg;
        }

        public static Tensor<int> Range(Scalar<int> stop)
        {
            return new Range(0, stop);
        }

        public static Tensor<int> Neq(Tensor<int> x, Tensor<int> y)
        {
            return new Neq<Tensor<int>, int>(x, y);
        }

        public static Tensor<T> Sign<T>(Tensor<T> x) => Apply(x, _x => Sign(_x));

        public static Tensor<T> ReLu<T>(Tensor<T> x) => Apply(x, _x => ReLu(_x));

        public static Tensor<T> Apply<T>(Tensor<T> x, Func<Scalar<T>, Scalar<T>> f)
        {
            return Tensor<T>.Elementwise.Create(x, f);
        }

        public static Tensor<T> Apply<T>(Tensor<T> x, Tensor<T> y, Func<Scalar<T>, Scalar<T>, Scalar<T>> f)
        {
            return Tensor<T>.Elementwise.CreateBinary(x, y, f);
        }

        public static Tensor<T> Apply<T>(Tensor<T> x, Tensor<T> y, Tensor<T> z, Func<Scalar<T>, Scalar<T>, Scalar<T>, Scalar<T>> f)
        {
            return Tensor<T>.Elementwise.CreateTernary(x, y, z, f);
        }


        public static Tensor<float> ConvolveSentence(Tensor<float> M, Tensor<float> K)
        {
            M.AssertOfDim(2); K.AssertOfDim(3);

            return Operators.FloatTensors.ConvolveCustom.Create(M, K,
                (m, k) => Dot(m, k),
                (d, k) => Dot(d, k, transposeY: true),
                (d, m) => Outer(m, d),
                M.Shape[0],
                "ij,jk->ik"
            );
        }

        public static Tensor<float> ConvolveCustom(Tensor<float> M, Tensor<float> K, string einstein)
        {
            var xyz = EinsteinSumTools.EinsteinSplit(einstein);

            var zyx = $"{xyz.Item3},{xyz.Item2}->{xyz.Item1}";
            var zxy = $"{xyz.Item3},{xyz.Item1}->{xyz.Item2}";

            return Operators.FloatTensors.ConvolveCustom.Create(M, K,
                (m, k) => EinsteinSum(m, k, einstein),
                (d, k) => EinsteinSum(d, k, zyx),
                (d, m) => EinsteinSum(d, m, zxy),
                M.Shape[0],
                einstein
            );
        }

        public static Tensor<float> CorrelateCustom(Tensor<float> M, Tensor<float> K, string einstein)
        {
            return Operators.FloatTensors.CorrelateCustom.Create(M, K,
                (m, k) => EinsteinSum(m, k, einstein),
                M.Shape[0],
                einstein
            );
        }

        public static Tensor<float> Outer(Tensor<float> x, Tensor<float> y) =>
            Operators.FloatTensors.TensorDot.Create(x, EmptyArray<int>.Value, y, EmptyArray<int>.Value);

        public static Tensor<T> Concat<T>(int axis, params Tensor<T>[] inputs) =>
            TheaNet.Operators.Tensors.Concat<T>.Create(axis, inputs);

        public static Tensor<float> EinsteinSum(Tensor<float> x, Tensor<float> y, string einstein) =>
            EinsteinSum<float>.Create(x, y, einstein);

        public static Tensor<float> BatchNormalization(Tensor<float> x, int axis, float epsilon = 0.0001f)
        {
            var mean = Mean(x, axis, keepDims: true);
            var variance = Mean(Square(x - axis));
            return (x - mean) / Sqrt(variance + epsilon);
        }

        public static Tensor<float> Switch(Tensor<int> mask, Tensor<float> ifTrue, Tensor<float> ifFalse) =>
            Apply(mask.As<float>(), ifTrue, ifFalse, (m, t, f) => Switch(m, t, f));

        public static Tensor<int> Switch(Tensor<int> mask, Tensor<int> ifTrue, Tensor<int> ifFalse) =>
            Apply(mask, ifTrue, ifFalse, (m, t, f) => Switch(m, t, f));
    }
}
