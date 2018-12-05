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
using System.Collections.Specialized;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Proxem.NumNet;
using Proxem.NumNet.Single;
using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Samples
{
    using static Slicer;

    public class Tsne
    {
        public readonly Tensor<float>.Shared X, Y, P;
        private Tensor<float>.Shared mask, YMomentum, dYLast;
        public readonly Scalar<float> KL_Loss;
        public readonly Func<Array<float>> dY;
        public readonly Func<float> Loss;
        public readonly Action Train;

        public Tsne(Array<float> X_, int dims, float perplexity)
        {
            X_.AssertOfDim(2);
            int n = X_.Shape[0];

            X = T.Shared(X_, "X");
            Y = T.Shared(NN.Random.Uniform(-1f, 1f, n, dims), "Y");

            YMomentum = T.Shared(NN.Zeros(n, dims), "YMomentum");
            dYLast = T.Shared(NN.Zeros(n, dims), "dYLast");

            // ones everywhere, zero on the diag
            mask = T.Shared(NN.Ones(n, n) - NN.Eye(n), "mask");

            // Compute pairwise affinities
            var sum_Y = T.Sum(Y * Y, 1, keepDims: true);

            var num = 1 / (1 - T.DimShuffle((2 * T.Dot(Y, Y, transposeY: true) + sum_Y), 1, 0) + sum_Y);
            // set the diag to zero
            num *= mask;

            var Q = num / T.Sum(num);
            //Q = T.Max(Q, 1e-12f);

            var P_ = x2p(X_, 1e-5f, perplexity);
            P_ = P_ * 4f; // early exaggeration
            P_ = NN.Apply(P_, x => Math.Max(x, 1e-12f));
            P = T.Shared(P_, "P");

            KL_Loss = T.Sum(P * T.Log(P / Q));

            dY = T.Function(output: T.Grad(KL_Loss, Y));
            Loss = T.Function(output: KL_Loss);

            var updates = MomentumUpdate(Y, YMomentum, dYLast, T.Grad(KL_Loss, Y), 500);
            Train = T.Function(updates);
        }

        public Array<float> x2p(Array<float> X, float tol = 1e-5f, float perplexity = 30f, bool sym = true, bool normalize = true)
        {
            //"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

            // Initialize some variables
            Console.WriteLine("Computing pairwise distances...");
            int n = X.Shape[0], d = X.Shape[1];
            var sum_X = NN.Sum(X * X, axis: 1);
            var D = (-2 * X.Dot(X.T) + sum_X).T + sum_X;
            var P = NN.Zeros(n, n);
            var beta = NN.Ones(n);
            var logU = (float)Math.Log(perplexity);
            var Di = NN.Zeros(n, n - 1);

            // Loop over all datapoints
            for (int i = 0; i < n; ++i)
            {
                // Print progress
                if (i % 500 == 0)
                    Console.WriteLine("Computing P-values for point {0} of {1} ...", i, n);

                // Compute the Gaussian kernel and entropy for the current precision
                var betamin = float.NegativeInfinity;
                var betamax = float.PositiveInfinity;
                Di[i, Until(i)] = D[Until(i)];
                if(i + 1 < n)
                    Di[i, From(i + 1)] = D[From(i)];

                var H_thisP = Hbeta(Di, beta.Item[i]);
                var H = H_thisP.Item1; var thisP = H_thisP.Item2;

                // Evaluate whether the perplexity is within tolerance
                var Hdiff = H - logU;
                var tries = 0;
                while (Math.Abs(Hdiff) > tol && tries < 50)
                {
                    // If not, increase or decrease precision
                    if (Hdiff > 0)
                    {
                        betamin = beta.Item[i];
                        if (float.IsInfinity(betamax))
                            beta.Item[i] = beta.Item[i] * 2;
                        else
                            beta.Item[i] = (beta.Item[i] + betamax) / 2;
                    }
                    else
                    {
                        betamax = beta.Item[i];
                        if (float.IsInfinity(betamin))
                            beta.Item[i] = beta.Item[i] / 2;
                        else
                            beta.Item[i] = (beta.Item[i] + betamin) / 2;
                    }
                    // Recompute the values
                    H_thisP = Hbeta(Di, beta.Item[i]);
                    H = H_thisP.Item1; thisP = H_thisP.Item2;

                    Hdiff = H - logU;
                    tries = tries + 1;
                }

                // Set the final row of P
                P[i, Until(i)] = thisP[Until(i)];
                if(i + 1 < n)
                    P[i, From(i + 1)] = thisP[From(i)];
            }
            var sigma = NN.Mean(NN.Sqrt(1 / beta));
            Console.WriteLine("Mean value of sigma: {0}", sigma);

            // Return final P-matrix
            if (sym)
                P += P.T;
            if (normalize)
                P /= NN.Sum(P);
            return P;
        }

        /// <summary>
        /// Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution.
        /// </summary>
        public Tuple<float, Array<float>> Hbeta(Array<float> D, float beta = 1.0f)
        {
            // Compute P-row and corresponding perplexity
            var P = NN.Exp(-beta * D);
            var sumP = NN.Sum(P);
            var H = (float) Math.Log(sumP) + beta * NN.Sum(D * P) / sumP;
            P.Scale(1 / sumP, result: P);
            return Tuple.Create(H, P);
        }

        public OrderedDictionary MomentumUpdate(Tensor<float>.Shared W, Tensor<float>.Shared gains, Tensor<float>.Shared iW, Tensor<float> dW, float lr, float rho=0.8f, OrderedDictionary dic = null)
        {
            dic = dic ?? new OrderedDictionary();

            var gains2 = (gains + 0.2f) * T.Eq(dW > 0f, iW > 0f) + (gains * 0.8f) * T.Neq((dW > 0f), (iW > 0f));
            gains2 = T.Clip(gains2, 0.01f, 10.0f);

            var iW2 = rho * iW + lr * (gains2 * dW);
            dic[W] = W - iW2;
            dic[iW] = iW2;
            dic[gains] = gains2;

            return dic;
        }
    }
}
