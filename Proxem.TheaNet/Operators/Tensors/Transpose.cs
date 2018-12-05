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

namespace Proxem.TheaNet.Operators.Tensors
{
    using Dim = Scalar<int>;
    using Perm = XList<Scalar<int>, int>;

    /// <summary>
    /// A view on a tensor with permuted dimensions (aka DimShuffle)
    /// </summary>
    public class Transpose<T> : Tensor<T>.NAry
    {
        public readonly int[] Permutation;

        public static Tensor<T> Create(Tensor<T> x, int[] perm)
        {
            x.AssertOfDim(perm.Length);
            if (x is Transpose<T> transpose)
            {
                x = transpose.x;
                perm = Compose(perm, transpose.Permutation);
            }

            perm.Apply(a => a < 0 ? a + x.NDim : a, result: perm);
            return IsIdentity(perm) ? x : new Transpose<T>(x, perm, ToAxes(perm));
        }

        private Transpose(Tensor<T> x, int[] perm, Perm perm1) : base("DimShuffle", x, perm1)
        {
            this.Permutation = perm;
            Shape = new Dim[perm.Length];
            perm.Apply(a => x.Shape[a], Shape);
        }

        private static int[] ToInt(Perm perm) => perm.Select(a => ((Scalar<int>.Const)a).Value).ToArray();
        private static Perm ToAxes(int[] perm) => new Perm(perm.Select(a => Op.Const(a)).ToArray());

        public static bool IsIdentity(int[] perm)
        {
            for (int i = 0; i < perm.Length; ++i)
                if (perm[i] != i)
                    return false;
            return true;
        }

        public int[] ReversePerm()
        {
            var reversed = new int[Permutation.Length];
            for (int i = 0; i < Permutation.Length; ++i)
                reversed[Permutation[i]] = i;
            return reversed;
        }

        static int[] Compose(int[] permA, int[] permB) => permB.Apply(a => permA[a]);

        public override sealed Dim[] Shape { get; }

        public Tensor<T> x => (Tensor<T>)Inputs[0];

        public override void Backward(Tensor<T> delta, Backpropagation bp)
        {
            bp.PushGradientTo(x, delta.DimShuffle(ReversePerm()));
        }

        public override NAry Clone(IReadOnlyList<IExpr> inputs) =>
            new Transpose<T>((Tensor<T>)inputs[0], ToInt((Perm)inputs[1]), (Perm)inputs[1]);
    }
}
