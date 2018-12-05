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

namespace Proxem.TheaNet
{
    using NumNet;
    using static BlocksTag;

    public static class BlocksTag
    {
        public const string EMBEDDINGS = "embeddings";
        public const string LINEAR = "linear";
        public const string BIAS = "bias";
        public const string SCALING = "scaling";
        public const string KERNEL = "kernel";
        public const string GATE = "gate";
    }

    public static class Blocks
    {
        public delegate Tensor<float> Block(Tensor<float> x);

        public static Tensor<float> Linear(string name, Tensor<float> x, int output, float scale = -1f, bool bias = true)
        {
            var input = Get(x.Shape[x.NDim - 1]);
            float s = scale >= 0 ? scale : (float)Math.Sqrt(6f / (input + output));
            var W = NN.Random.Uniform(-s, s, input, output);
            return Linear(name, x, W, bias);
        }

        public static Tensor<float> Linear(string name, Tensor<float> x, Array<float> linear, bool bias = true)
        {
            var W = Op.Shared(linear, name);
            W.Tag(LINEAR);

            var y = Op.Dot(x, W);
            if (bias)
            {
                var b_shape = new int[y.NDim];
                for (int i = 0; i < y.NDim - 1; ++i) b_shape[i] = 1;
                b_shape[y.NDim - 1] = Get(y.Shape[y.NDim - 1]);

                var b = Op.Shared(NN.Zeros(b_shape), name + "_bias");
                b.Tag(BIAS);

                y += b;
            }
            y.Name = name + "_out";
            return y;
        }

        public static Tensor<float> Id(Tensor<float> x) => x;

        public static Tensor<float> MLP(string name, Tensor<float> x, int[] sizes, Block[] activations, float scale = -1f, bool bias = true)
        {
            if (sizes.Length != activations.Length)
                throw new ArgumentException("need as many sizes than activations");
            if (sizes.Length == 1)
                return activations[0](Linear(name, x, sizes[0], scale, bias));

            for (int i = 0; i < sizes.Length; ++i)
            {
                x = Linear(name + "_" + i, x, sizes[i], scale, bias);
                x = (activations[i] ?? Id)(x);
            }
            x.Name = name + "_out";
            return x;
        }

        public static Tensor<float> Conv(string name, Tensor<float> x, int output, int kernelSize, float scale = -1f, Block pooling = null, bool bias = true)
        {
            var inputDim = Get(x.Shape[x.NDim - 1]);
            float s = scale >= 0 ? scale : (float)Math.Sqrt(6f / (inputDim * kernelSize + output));
            return Conv(name, x, NN.Random.Uniform(-s, s, kernelSize, inputDim, output), pooling: pooling, bias: bias);
        }

        public static Tensor<float> Conv(string name, Tensor<float> x, Array<float> kernel, Block pooling = null, bool bias = true)
        {
            var einstein = "i,io->o";
            if (x.NDim == 3) einstein = "bi,io->bo";

            var W = Op.Shared(kernel, name);
            W.Tag(KERNEL); W.Tag(name);

            var y = Op.ConvolveCustom(x, W, einstein);
            if (pooling != null) y = pooling(y);

            if (bias)
            {
                var b_shape = new int[y.NDim];
                for (int i = 0; i < y.NDim - 1; ++i) b_shape[i] = 1;
                b_shape[y.NDim - 1] = kernel.Shape[kernel.NDim - 1];
                var b = Op.Shared(NN.Zeros(b_shape), name + "_bias");
                //var b = Op.Shared(NN.Random.Uniform(-s, s, b_shape), name + "_bias");
                b.Tag(BIAS); b.Tag(name);

                y += b;
            }

            y.Name = name + "_out";
            return y;
        }

        public static Tensor<float> Embeddings(string name, Tensor<int> ids, int vocSize, int dim, float scale = -1f)
        {
            scale = scale >= 0 ? scale : (float)Math.Sqrt(3f / dim);
            return Embeddings(name, ids, NN.Random.Uniform(-scale, scale, vocSize, dim));
        }

        public static Tensor<float> Embeddings(string name, Tensor<int> ids, Array<float> L)
        {
            var L_ = Op.Shared(L, name);
            L_.Tag(EMBEDDINGS);
            var y = L_[ids];

            y.Name = name + "_out";
            return y;
        }

        //private static int GetOr(Scalar<int> value, int or = 50) =>
        //    (value as Scalar<int>.Const)?.Value ?? or;

        private static int Get(Scalar<int> value)
        {
            if (value is Scalar<int>.Const c) return c.Value;
            throw new RankException($"Can't create shared with dim {value}.");
        }
    }
}
