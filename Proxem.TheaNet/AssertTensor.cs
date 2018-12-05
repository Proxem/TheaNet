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
using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet
{
    public static class AssertTensor
    {
        public static void AssertOfDim(this ITensor thiz, int n)
        {
            if (thiz.NDim != n)
                throw RankException("{0} should be of dim: {1} but is of dim: {2}", thiz.ToString(), n, thiz.NDim);
        }

        public static void BindToShape(this ITensor thiz, ITensor that)
        {
            thiz.BindToShape(that.Shape);
        }

        public static void AssertOfShape<T>(this ITensor<T> thiz, ITensor that)
        {
            thiz.AssertOfShape(that.Shape);
        }

        public static void AssertOfShape<T>(this ITensor<T> thiz, params Scalar<int>[] shape)
        {
            var a = thiz.Shape;
            if (thiz.NDim != shape.Length)
                throw RankException("{0} of shape {1}, won't match with: {2}", thiz, thiz.Shape.Format(thiz), shape.Format(thiz));

            for (int d = 0; d < thiz.NDim; ++d)
                if (!ShapeExtension.CanEqualTo(a[d], shape[d]))
                    throw RankException("{0} of shape {1}, won't match with: {2}", thiz, thiz.Shape.Format(thiz), shape.Format(thiz));
        }

        public static void BindToShape(this ITensor thiz, Scalar<int>[] shape)
        {
            var a = thiz.Shape;
            thiz.AssertOfDim(shape.Length);
            for (int d = 0; d < thiz.NDim; ++d)
                ShapeExtension.Bind(ref a[d], ref shape[d]);
        }

        public static RankException RankException(string format, params object[] o)
        {
            return new RankException(string.Format(format, o));
        }

        /// <summary>
        /// Checks the gradient of an expression with one input.
        /// If a shape of the input is unknown, it will be replaced by 10.
        /// </summary>
        public static void PassesGradientCheck<X>(Tensor<X>.Var input, Scalar<float> expr, Tensor<float> W,
            float epsilon=0.001f, float relativeErr=1e-3f, float absErr=1e-4f, int repeat = 50, Func<Array<X>> init = null)
        {
            var xShape = input.Shape.Select(s => (s as Scalar<int>.Const)?.Value ?? 10).ToArray();
            var checkGrad = T.RandomGradientCheck(new[] { input }, expr, W);

            if (init == null)
                init = () => NN.Random.Uniform(-1f, 1f, xShape).As<X>();

            var fault = 0;
            var last = "";
            for (int _ = 0; _ < repeat; ++_)
            {
                var x = init();
                var checkRes = checkGrad(x, epsilon);
                var finite = checkRes.Item1;
                var backpropagated = checkRes.Item2;

                if (!AssertArray.CheckAreAlmostEqual(finite, backpropagated, relativeErr, absErr))
                {
                    var abs = Math.Abs(finite - backpropagated);
                    var relative = 2 * abs / (Math.Abs(finite) + Math.Abs(backpropagated));
                    last += $"Expected: {finite}, actual {backpropagated}, diff {abs}, relative {relative}.\n";
                    ++fault;
                }
            }

            if(fault > 0)
                throw new Exception($"The computed gradient of {W.Name} doesn't match finite difference (failed {fault} times over {repeat}).\n{last}");
        }

        /// <summary>
        /// Checks the gradient of an expression without inputs.
        /// </summary>
        public static void PassesGradientCheck(Scalar<float> expr, Tensor<float> W,
            float epsilon = 0.001f, float relativeErr = 1e-3f, float absErr = 1e-4f, int repeat = 50)
        {
            var checkGrad = T.RandomGradientCheck(EmptyArray<IVar>.Value, expr, W);

            var fault = 0;
            var last = "";
            for (int _ = 0; _ < repeat; ++_)
            {
                var checkRes = checkGrad(epsilon);
                var finite = checkRes.Item1;
                var backpropagated = checkRes.Item2;

                if (!AssertArray.CheckAreAlmostEqual(finite, backpropagated, relativeErr, absErr))
                {
                    var abs = Math.Abs(finite - backpropagated);
                    var relative = 2 * abs / (Math.Abs(finite) + Math.Abs(backpropagated));
                    last += $"Expected: {finite}, actual {backpropagated}, diff {abs}, relative {relative}.\n";
                    ++fault;
                }
            }

            if (fault > 0)
                throw new Exception($"The computed gradient of {W.ToString()} doesn't match finite difference (failed {fault} times over {repeat}).\n{last}");
        }

        /// <summary>
        /// Checks the gradient of an expression without inputs.
        /// </summary>
        public static void PassesGradientCheck(Scalar<float> expr, Scalar<float> W,
            float epsilon = 0.001f, float relativeErr = 1e-3f, float absErr = 1e-4f, int repeat = 6)
        {
            var checkGrad = T.RandomGradientCheck(EmptyArray<IVar>.Value, expr, W);
            var fault = 0;
            var errors = "";

            for (int _ = 0; _ < repeat; ++_)
            {
                var eps = (_ % 2 == 0) ? epsilon : -epsilon;
                var checkRes = checkGrad(eps);
                var finite = checkRes.Item1;
                var backpropagated = checkRes.Item2;

                if (!AssertArray.CheckAreAlmostEqual(finite, backpropagated, relativeErr, absErr))
                {
                    var abs = Math.Abs(finite - backpropagated);
                    var relative = 2 * abs / (Math.Abs(finite) + Math.Abs(backpropagated));
                    errors += $"For epsilon {eps} expected: {finite}, actual {backpropagated}, diff {abs}, relative {relative}.\n";
                    ++fault;
                }

                if(_ % 2 == 1)
                    epsilon *= 10;
            }

            if (fault > 0)
                throw new Exception($"The computed gradient of {W.ToString()} doesn't match finite difference (failed {fault} times over {repeat}).\n{errors}");
        }
    }
}
