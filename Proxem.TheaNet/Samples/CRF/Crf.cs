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
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Proxem.NumNet;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Samples.CRF
{
    public class Crf
    {
        /// <summary>
        /// Compute log(sum(exp(x), axis=axis) in a numerically stable fashion.
        /// </summary>
        /// <param name="x">A Theano tensor (any dimension will do).</param>
        /// <param name="axis">int or symbolic integer scalar, or None. Axis over which to perform the summation. `None`, the
        /// default, performs over all axes.</param>
        /// <returns>The result of the log(sum(exp(...))) operation.</returns>
        public static Tensor<float> LogSumExp(Tensor<float> x, int axis)
        {
            var xmax = T.Max(x, axis: axis, keepDims: true);
            var xmax_ = T.Max(x, axis: axis);
            return xmax_ + T.Log(T.Sum(T.Exp(x - xmax), axis: axis));
        }

        /// <summary>
        /// Given (symbolic) log-domain potentials, construct the graph for forward inference in a chain CRF.
        /// </summary>
        /// <param name="obs_potentials">(n_steps, n_classes) Axes correspond to time and the value of the discrete label variable
        /// This is the energy assigned to a configuration (so higher energy = lower probability).</param>
        /// <param name="chain_potentials">(n_classes, n_classes, n_classes) Axes correspond to left label state, right label state, and the global label.
        /// Corresponds to the energy of a given pair of labels adjacent to one another (higher energy = lower probability).</param>
        /// <param name="viterbi">Perform MAP inference with the Viterbi algorithm rather than marginalizing the step-specific
        /// label variables, Instead, use the single most likely configuration.</param>
        /// <returns>(1-dimensional) The energy assigned for a given global label.
        /// This can be turned into a log probability by subtracting logsumexp(energy).</returns>
        public static Tensor<float> Forward(Tensor<float> obs_potentials, Tensor<float> chain_potentials, bool viterbi = false)
        {
            Func<Tensor<float>, Tensor<float>, Tensor<float>> inner_function = (obs, prior_result/*, chain_potentials*/) =>
            {
                prior_result = prior_result.DimShuffle(0, 'x', 1);
                obs = obs.DimShuffle('x', 0, 'x');
                if (viterbi)
                    return T.Max((-prior_result - obs - chain_potentials), axis: 0);
                else
                    return LogSumExp(-prior_result - obs - chain_potentials, axis: 0);
            };

            Debug.Assert(obs_potentials.NDim == 2);
            Debug.Assert(chain_potentials.NDim == 3);
            var initial = (obs_potentials[0].DimShuffle(0, 'x') * T.OnesLike(chain_potentials[0]));
            var scanned = T.Scan(
                fn: inner_function,
                outputsInfo: initial,
                sequences: new[] { obs_potentials[XSlicer.From(1)] }
                //non_sequences: chain_potentials
            );

            if (viterbi)
                return -(T.Max(scanned[-1], axis: 0));
            else
                return -LogSumExp(scanned[-1], axis: 0);
        }
    }
}
