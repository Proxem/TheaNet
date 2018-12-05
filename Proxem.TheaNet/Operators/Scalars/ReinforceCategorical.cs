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

namespace Proxem.TheaNet.Operators.Scalars
{
    /// <summary>
    /// **Experimental**: transforms a reward into a gradient.
    /// see: http://torch.ch/blog/2015/09/21/rmva.html#reinforce-algorithm
    /// </summary>
    public class ReinforceCategorical : Scalar<int>.NAry
    {
        public Scalar<float> Reward;
        public Scalar<float>.Shared Baseline;
        public readonly float Decay;
        private Tensor<float> Distribution, DistributionGradient;

        public static ReinforceCategorical Create(Tensor<float> x, string name, float decay = 0.9f)
        {
            x.AssertOfDim(1);
            var b = Op.Shared(0f, name);
            return new ReinforceCategorical(x, null, b, decay);
        }

        private ReinforceCategorical(Tensor<float> distribution, Scalar<float> reward, Scalar<float>.Shared baseline, float decay) : base("ReinforceCategorical", distribution, baseline)
        {
            Reward = reward;
            Baseline = baseline;
            Decay = decay;
            Distribution = distribution;
        }

        public override void Backward(Scalar<int> delta, Backpropagation bp)
        {
            if(DistributionGradient == null)
            {
                if (Reward == null)
                    throw new Exception($"No Reward was provided for ReinforceCategorical. Can't Backward");

                DistributionGradient = delta.As<float>() * (Reward - Baseline) / Distribution.Item[this] * Op.OneHot<float>(Distribution.Shape, new[] { this }, 1f);
            }
            bp.PushGradientTo(target: Distribution, delta: DistributionGradient);
            bp.PushGradientTo(target: Baseline, delta: (Baseline - Reward));
        }

        public override Scalar<int> Clone(IReadOnlyList<IExpr> inputs)
        {
            throw new NotImplementedException();
        }
    }
}
