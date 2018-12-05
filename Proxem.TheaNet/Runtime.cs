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
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;

using Proxem.NumNet;
using Proxem.NumNet.Int32;
using Proxem.NumNet.Single;

namespace Proxem.TheaNet
{
    public class Runtime
    {
        // TODO: using static NN;
        // a lot of methods come directly from NumNet.NN, maybe we should just import NN alongside with Runtime to eliminate redundant code

        public static IDictionary<string, float> Float = new Dictionary<string, float>();
        public static IDictionary<string, int> Int = new Dictionary<string, int>();
        public static IDictionary<string, Array<float>> FloatArray = new GuardedDictionary<float>();
        public static IDictionary<string, Array<int>> IntArray = new GuardedDictionary<int>();
        public IDictionary<string, Delegate> CustomFunctions = new Dictionary<string, Delegate>();

        public static void Reset()
        {
            Float.Clear();
            Int.Clear();
            FloatArray.Clear();
            IntArray.Clear();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Exp(float x) => (float)Math.Exp(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Log(float x)
        {
            if (x <= 0) return -1e15f;
            return (float)Math.Log(x);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Pow(float x, float y) => (float)Math.Pow(x, y);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Tanh(float x) => (float)Math.Tanh(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ReLu(float x) => x > 0 ? x : 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sqrt(float x)
        {
            if (x < 0) throw new Exception("Sqrt < 0");
            return (float)Math.Sqrt(x);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Max(int x, int y) => Math.Max(x, y);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Max(float x, float y) => Math.Max(x, y);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> Max(Array<float> a, int axis, bool keepDims = true, Array<float> result = null) => a.Max(axis, keepDims, result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Min(int x, int y) => Math.Min(x, y);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Min(float x, float y) => Math.Min(x, y);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sum(Array<float> a) => a.Sum();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> Sum(Array<float> a, int axis, bool keepDims = true, Array<float> result = null) => a.Sum(axis, keepDims: keepDims, result: result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sum(Array<int> a) => a.Sum();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<int> Sum(Array<int> a, int axis, bool keepDims = true, Array<int> result = null) => a.Sum(axis, keepDims: keepDims, result: result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Norm2(Array<float> a) => NN.Norm2(a);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Square(float x) => x * x;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Square(int x) => x * x;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<T> Square<T>(Array<T> x) => x * x;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Abs(float x) => Math.Abs(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(float x) => (float)(1 / (1 + Math.Exp(-x)));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Clip(float x, float min, float max)
        {
            if (x < min) return min;
            if (x > max) return max;
            return x;
        }

        // HACK: we should pass a second buffer to softmax
        [ThreadStatic]
        private static ResizableArray<float> _softmaxBuffer1;
        [ThreadStatic]
        private static ResizableArray<float> _softmaxBuffer2;
        [ThreadStatic]
        private static ResizableArray<float> _softmaxBuffer3;

        private static ResizableArray<float> _softmaxBuffer(int ndim)
        {
            switch (ndim)
            {
                case 1:
                    if (_softmaxBuffer1 == null) _softmaxBuffer1 = new ResizableArray<float>(-1);
                    return _softmaxBuffer1;
                case 2:
                    if (_softmaxBuffer2 == null) _softmaxBuffer2 = new ResizableArray<float>(-1, -1);
                    return _softmaxBuffer2;
                default:
                    if (_softmaxBuffer3 == null) _softmaxBuffer3 = new ResizableArray<float>(-1, -1, -1);
                    return _softmaxBuffer3;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> Softmax(Array<float> a, int axis = -1, Array<float> result = null)
        {
            axis = axis < 0 ? axis + a.NDim : axis;
            // reuse the shape of a to avoid creating a temporary array
            var shape = a.Shape;
            int save = shape[axis];
            shape[axis] = 1;
            var buff = _softmaxBuffer(a.NDim).ResizeTo(shape);
            // restore the shape of a
            shape[axis] = save;

            return NN.Softmax(a, axis, result, buff);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<T> Const<T>(T a, int[] shape, Array<T> result = null)
        {
            if (result != null)
            {
                result.FillWith(a);
                return result;
            }
            else return NN.Const(a, shape);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> LogSumExp(Array<float> a, int axis = -1, bool keepDims = true, Array<float> result = null) => NN.LogSumExp(a, axis, keepDims, result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> Uniform(float min, float max, int[] shape, Array<float> result = null) =>
            result == null ? NN.Random.Uniform(min, max, shape) : NN.Random.Uniform(min, max, result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<int> Uniform(int min, int max, int[] shape, Array<int> result = null) =>
            result == null ? NN.Random.Uniform<int>(min, max, shape) : NN.Random.Uniform(min, max, result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> Normal(float mean, float std, int[] shape, Array<float> result = null) =>
            result == null ? NN.Random.Normal(mean, std, shape) : NN.Random.Normal(mean, std, result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ReinforceCategorical(Array<float> distribution, float baseline) => NN.Random.Multinomial(distribution);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> OneHot(int[] shape, int y, float value = 1, Array<float> result = null)
        {
            if (result != null)
                result.Clear();
            else
                result = NN.Zeros<float>(shape);
            result.Item[y] = value;
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> OneHotPoint(int[] shape, int[] y, float value = 1, Array<float> result = null)
        {
            if (result != null)
                result.Clear();
            else
                result = NN.Zeros<float>(shape);
            result.Item[y] = value;
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<T> OneHot<T>(int[] shape, int y, Array<T> value, Array<T> result = null)
        {
            if (result != null)
                result.FillWith(Numeric<T>.Zero);
            else
                result = NN.Zeros<T>(shape);
            result[y] = value;
            return result;
        }

        /// <summary>
        ///  Creates an empty x array and fill a part of it: x[slices] = value
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="slices">the slices to fill</param>
        /// <param name="shape">the shape of x</param>
        /// <param name="value">the values to fill with</param>
        /// <param name="result"> resulting array, if null it will create one.</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<T> OneHot<T>(int[] shape, Slice[] slices, Array<T> value, Array<T> result = null)
        {
            if (result != null)
                result.FillWith(Numeric<T>.Zero);
            else
                result = NN.Zeros<T>(shape);
            result[slices] = value;
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int[] Concatenate(int s1, int[] shape)
        {
            var result = new int[shape.Length + 1];
            result[0] = s1;
            Array.Copy(shape, 0, result, 1, shape.Length);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> Deindex(Array<float> a, int[] shape, Array<int>[] indexArrays, Array<float> result = null)
        {
            // see Array<Type>.this[params Array<int>[] indices]
            result = result?.Clear() ?? NN.Zeros<float>(shape);
            var firstArray = indexArrays[0];

            if (firstArray.Shape.Length == 1)
            {
                var indices = new int[indexArrays.Length];
                for (int axis0 = 0; axis0 < firstArray.Shape[0]; axis0++)
                {
                    for (int i = 0; i < indices.Length; i++)
                        indices[i] = indexArrays[i].Item[axis0];
                    result[indices].Acc(a[axis0]);
                    //result.Item[indices] += a.Item[axis0];
                }
                return result;
            }
            else if (firstArray.Shape.Length == 2)
            {
                var indices = new int[indexArrays.Length];
                for (int axis0 = 0; axis0 < firstArray.Shape[0]; axis0++)
                {
                    for (int axis1 = 0; axis1 < firstArray.Shape[1]; axis1++)
                    {
                        for (int i = 0; i < indices.Length; i++)
                            indices[i] = indexArrays[i].Item[axis0, axis1];
                        result[indices].Acc(a[axis0, axis1]);
                    }
                }
                return result;
            }

            throw new NotImplementedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> Dispatch(float a, int[] shape, Array<int>[] indexArrays, Array<float> result = null)
        {
            // see Array<Type>.this[params Array<int>[] indices]
            result = result?.Clear() ?? NN.Zeros<float>(shape);
            var firstArray = indexArrays[0];
            var dimMatches = indexArrays.Length == shape.Length;

            if (firstArray.NDim == 1)
            {
                var indices = new int[indexArrays.Length];
                for (int axis0 = 0; axis0 < firstArray.Shape[0]; axis0++)
                {
                    for (int i = 0; i < indices.Length; i++)
                        indices[i] = indexArrays[i].Item[axis0];
                    if (dimMatches)
                        result.Item[indices] += a;
                    else
                        result[indices] += a;
                }
                return result;
            }
            else if (firstArray.NDim == 2)
            {
                var indices = new int[indexArrays.Length];
                for (int axis0 = 0; axis0 < firstArray.Shape[0]; axis0++)
                    for (int axis1 = 0; axis1 < firstArray.Shape[1]; axis1++)
                    {
                        for (int i = 0; i < indices.Length; i++)
                            indices[i] = indexArrays[i].Item[axis0, axis1];
                        if (dimMatches)
                            result.Item[indices] += a;
                        else
                            result[indices] += a;
                    }
                return result;
            }

            throw new NotImplementedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<T> DimShuffle<T>(Array<T> a, int[] axesPerm, Array<T> result = null)
        {
            return a.Transpose(axesPerm);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<T> ShapePadLeft<T>(Array<T> a) => a.Reshape(Concatenate(1, a.Shape));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> Outer(Array<float> a, Array<float> b) => a.Outer(b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> TensorDot(Array<float> a, IList<int> axesA, Array<float> b, IList<int> axesB, Array<float> result = null) => NN.TensorDot(a, axesA, b, axesB, result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<int> Range(int start, int stop, int step = 1, Array<int> result = null) => NN.Range(start, stop, step, result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> Dot(Array<float> a, Array<float> b, Array<float> result = null, float beta = 0, float alpha = 1, bool transA = false, bool transB = false) =>
            a.Dot(b, result, alpha, beta, transA, transB);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AssertOfDim<T>(Array<T> a, int dim) => a.AssertOfDim(dim);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<T> Insert<T>(Array<T> a, Array<T> other, int index, int axis, Array<T> result = null) => a.Insert(other, index, axis, result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> Convolve(Array<float> a, Array<float> kernel, Array<float> result = null, int mode = (int)ConvMode.Full) =>
            a.Convolve(kernel, result, (ConvMode)mode);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> Correlate(Array<float> a, Array<float> kernel, Array<float> result = null, int mode = (int)ConvMode.Full) =>
            a.Correlate(kernel, result, (ConvMode)mode);

        public static void Assert(bool cond)
        {
            if (!cond) throw new Exception("Assertion violation");
        }

        public static void AssertShapes(int[] a1, params int[] a2)
        {
            if (!System.Linq.Enumerable.SequenceEqual(a1, a2)) throw new Exception("Shape violation");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<T> Concat<T>(int axis, Array<T>[] inputs, Array<T> result = null)
        {
            if (result == null)
                return NN.Concat(axis, inputs);
            else
                return NN.Concat(axis, inputs, result);
        }

        public static T Print<T>(string format, T x)
        {
            if (format == null)
                Trace.WriteLine(x);
            else
                Trace.WriteLine(string.Format(format, x));
            return x;
        }

        public static T Print<T>(T x)
        {
            Trace.WriteLine(x);
            return x;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> ConvolveCustom(
            Array<float> x, Array<float> y, int[] shape,
            Func<Array<float>, Array<float>, Array<float>> f,
            Array<float> result = null
        ) =>
            x.ConvolveCustom(y, result?.Clear() ?? NN.Zeros(shape), f);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> CorrelateCustom(
            Array<float> x, Array<float> y, int[] shape,
            Func<Array<float>, Array<float>, Array<float>> f,
            Array<float> result = null
        ) =>
            x.CorrelateCustom(y, result?.Clear() ?? NN.Zeros(shape), f);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<int> Argmax(Array<float> x, int axis, bool keepDims = true, Array<int> result = null) => x.Argmax(axis, keepDims, result);

        public static int Argmax(Array<float> x) => x.Argmax();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> UnArgmax(Array<float> x, Array<int> argmax, int axisSize, int axis, bool keepDims = true, Array<float> result = null)
        {
            if (result != null) result.Clear();
            return NN.UnArgmax(x, argmax, axis, axisSize, keepDims, result);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> EinsteinSum(Array<float> x, Array<float> y, string einstein) =>
            NN.EinsteinSum(x, y, einstein);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Ge(float x, float y) => x >= y ? 1 : 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Gt(float x, float y) => x > y ? 1 : 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Ge(int x, int y) => x >= y ? 1 : 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Gt(int x, int y) => x > y ? 1 : 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<float> Broadcast(Array<float> x, int[] shape, Array<float> result = null) =>
            x.Add(NN.Zeros<float>(shape), result: result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<int> Broadcast(Array<int> x, int[] shape, Array<int> result = null) =>
            x.Add(NN.Zeros<int>(shape), result: result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<T> Copy<T>(Array<T> x, Array<T> result = null) => NN.Copy(x, result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Switch(float mask, float ifTrue, float ifFalse) =>
            mask > 0 ? ifTrue: ifFalse;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Switch(int mask, int ifTrue, int ifFalse) =>
            mask > 0 ? ifTrue : ifFalse;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<T> SliceAlong<T>(Array<T> x, int axis, int index)
        {
            if (axis == 0)
                return x[index];
            else
            {
                var slices = x.Slices();
                slices[axis] = index;
                return x[slices];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DirichletPdf(float alpha, Array<float> x) => NN.DirichletPdf(alpha, x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T Invoke<T>(string functionName, params object[] @params)
        {
            var function = CustomFunctions[functionName];
            return (T)function.Method.Invoke(function.Target, @params);
        }

        /// <summary>
        /// Wrapper around an array, which can easily scale up.
        /// </summary>
        public sealed class ResizableArray<T>
        {
            private Array<T> _content;
            private Array<T> _view;
            public readonly IReadOnlyList<int> ResizableAxes;

            /// <summary></summary>
            /// <param name="shape">Shape of the array. -1 marks axis that can be resized.</param>
            public ResizableArray(params int[] shape)
            {
                var axes = new List<int>();
                for (int i = 0; i < shape.Length; ++i)
                {
                    if (shape[i] < 0)
                    {
                        axes.Add(i);
                        shape[i] = 2;
                    }
                }
                _content = new Array<T>(shape);
                _view = new Array<T>(shape.CopyToNew(), _content.Values, 0, _content.Stride);
                ResizableAxes = axes;
            }

            public static implicit operator Array<T>(ResizableArray<T> resizable) => resizable._view;

            /// <summary>
            /// Returns an array of the given shape. Will alocate a new T[] if we ask for a bigger size thatn previously allocated.
            /// In case of resizing the previous values will be lost.
            /// </summary>
            /// <param name="shape">Only uses the dims for axes that are resizable.</param>
            /// <returns>A view of the requested shape.</returns>
            public Array<T> ResizeTo(params int[] shape)
            {
                bool needResize = false;
                for(int i = 0; i < ResizableAxes.Count; ++i)
                {
                    var a = ResizableAxes[i];
                    var d = shape[a];
                    if (d > _content.Shape[a])
                    {
                        _content.Shape[a] = d;
                        needResize = true;
                    }
                    if(d != _view.Shape[a])
                    {
                        _view.Shape[a] = d;
                    }
                }

                if (needResize)
                {
                    var size = 1;
                    foreach (var d in _content.Shape)
                        size *= d;
                    _content.Values = new T[size];
                    StridedExtension.ComputeStride(_content.Shape, result: _content.Stride);
                    _view.Values = _content.Values;
                }

                return _view;
            }
        }
    }

    /// <summary>
    /// Dictionary of Array. When overriding an array, checks that the number of dimension is stable.
    /// </summary>
    /// <typeparam name="T">The type of stored arrays.</typeparam>
    public class GuardedDictionary<T> : IDictionary<string, Array<T>>
    {
        private IDictionary<string, Array<T>> Dictionary = new Dictionary<string, Array<T>>();
        public Array<T> this[string key]
        {
            get
            {
                return Dictionary[key];
            }

            set
            {
                Array<T> oldValue;
                if (Dictionary.TryGetValue(key, out oldValue))
                {
                    if (value.Shape.Length != oldValue.Shape.Length)
                        throw new RankException($"Wrong number of dimensions: expected {oldValue.Shape.Length} got {value.Shape.Length} with shape ({string.Join(", ", value.Shape)}). Container named '{key}'");
                }
                var farray = value as Array<float>;
#if CHECK_NAN
                if (farray != null)
                {
                    if (float.IsNaN(farray.Sum())) throw new Exception("NaN");
                }
#endif
                Dictionary[key] = value;
            }
        }

        public int Count => Dictionary.Count;

        public bool IsReadOnly => Dictionary.IsReadOnly;

        public ICollection<string> Keys => Dictionary.Keys;

        public ICollection<Array<T>> Values => Dictionary.Values;

        public void Add(KeyValuePair<string, Array<T>> item) => Dictionary.Add(item);

        public void Add(string key, Array<T> value) => Dictionary.Add(key, value);

        public void Clear() => Dictionary.Clear();

        public bool Contains(KeyValuePair<string, Array<T>> item) => Dictionary.Contains(item);

        public bool ContainsKey(string key) => Dictionary.ContainsKey(key);

        public void CopyTo(KeyValuePair<string, Array<T>>[] array, int arrayIndex) => Dictionary.CopyTo(array, arrayIndex);

        public IEnumerator<KeyValuePair<string, Array<T>>> GetEnumerator() => Dictionary.GetEnumerator();

        public bool Remove(KeyValuePair<string, Array<T>> item) => Dictionary.Remove(item);

        public bool Remove(string key) => Dictionary.Remove(key);

        public bool TryGetValue(string key, out Array<T> value) => Dictionary.TryGetValue(key, out value);

        IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)Dictionary).GetEnumerator();
    }

    public static class Int32Extensions
    {
        public static Array<int> Range(this int start, int stop) => NN.Range(start, stop);
    }

    public static class ArrayExtensions
    {
        public static Array<T> OfShape<T>(this Array<T> t, params int[] shape)
        {
            t.AssertOfShape(shape);
            return t;
        }

        // inline expansion of params int[1]
        public static Array<T> OfShape<T>(this Array<T> t, int dim)
        {
            t.AssertOfShape(dim);
            return t;
        }

        // inline expansion of params int[2]
        public static Array<T> OfShape<T>(this Array<T> t, int dim1, int dim2)
        {
            t.AssertOfShape(dim1, dim2);
            return t;
        }

        // inline expansion of params int[3]
        public static Array<T> OfShape<T>(this Array<T> t, int dim1, int dim2, int dim3)
        {
            t.AssertOfShape(dim1, dim2, dim3);
            return t;
        }
    }
}
