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
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Proxem.NumNet;

namespace Proxem.TheaNet.Samples
{
    public class ReinforceMul
    {
        public readonly int N;
        public readonly Func<float, int> Train;
        public readonly Func<int, int, int> Mul;
        private Scalar<float>.Shared baseline;

        public ReinforceMul(int N = 7)
        {
            this.N = N;
            int dim0 = N, dim1 = N * N, dim2 = N;

            var a = Op.Scalar<int>("a");
            var b = Op.Scalar<int>("b");
            var c_gold = Op.Scalar<int>("c_gold");
            var L = Op.Shared(NN.Random.Uniform(-1f, 1f, N, dim0), "L");

            var xa = Blocks.Linear("Wa", L[a], dim1);
            var xb = Blocks.Linear("Wb", L[b], dim1);
            var xab = Blocks.Linear("Wab", L[a] * L[b], dim1);

            var x = Blocks.Linear("Wc", Op.Tanh(xa + xb + xab), dim2);
            var y = Op.Softmax(x);

            var c = Operators.Scalars.ReinforceCategorical.Create(y, "baseline");
            c.Name = nameof(c);
            var eq = Op.Eq(c_gold, c);
            c.Reward = eq.As<float>();
            c.Reward.Name = "reward";
            this.baseline = c.Baseline;

            var loss = -c.As<float>();
            var weights = loss.FindAll<Tensor<float>.Shared>();
            foreach (var W in weights)
                loss += 0.001f * Op.Norm2(W);
            loss.Name = nameof(loss);

            var grad = Op.Grad(loss); // reward isn't differentiable but Reinforce will still backpropagate gradients

            var lr = Op.Scalar<float>("lr");
            var updates = new OrderedDictionary();
            foreach (var W in grad.Keys)
                updates[W] = W - lr * grad[W];

            //var dB = (Scalar<float>)loss.Backpropagation.ScalarDerivatives[c.Baseline];
            //updates[c.Baseline] = c.Baseline - lr * dB;
            updates[c.Baseline] = c.Baseline * 0.9f + 0.1f * c.Reward;

            var train_ = Op.Function(new IVar[] { a, b, c_gold, lr }, c.Reward, updates);
            Train = lr_ => {
                var sample = NextSample();
                return (int)train_(sample.Item1, sample.Item2, sample.Item3, lr_);
            };

            Mul = Op.Function(input: (a, b), output: Op.Argmax(x));
        }

        public (int, int, int) NextSample()
        {
            var x = NN.Random.NextInt(N);
            var y = NN.Random.NextInt(N);
            return (x, y, (x * y) % N);
        }

        public void TestOn()
        {
            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    var c = Mul(i, j);
                    var c_gold = i * j % N;
                    if (c == c_gold)
                        Trace.Write($"{c}\t\t");
                    else
                        Trace.Write($"{c}({c_gold})\t");
                }

                Trace.WriteLine("");
            }
        }

        public float TrainFor(int epoch, int epochLength)
        {
            float res = 0f;
            for(int e = 0; e < epoch; ++e)
            {
                float lr = 0.1f / (1 + (float)Math.Sqrt(e));
                //float lr = 0.1f;
                int acc = 0;
                for(int i = 0; i < epochLength; ++i)
                    acc += Train(lr);
                res = (float)acc / epochLength;
                Trace.WriteLine($"End of epoch {e}, with lr {lr}. Accuracy: {100 * res}%. Baseline: {100 * baseline.Value}%.");
            }
            Trace.WriteLine($"Testing");
            TestOn();
            return res;
        }
    }

    public class ReinforceConst
    {
        public readonly int N, gold;
        public readonly Func<float, int> Train;
        public readonly Func<int, int> Const;

        public ReinforceConst(int N = 5)
        {
            this.N = N;
            gold = N / 2;

            var a = Op.Scalar<int>("a");
            var c_gold = Op.Scalar<int>("c_gold");
            var L = Op.Shared(NN.Random.Uniform(-1f, 1f, N, N), "L");

            var x = L[a];
            var y = Op.Softmax(x);

            var c = Operators.Scalars.ReinforceCategorical.Create(y, "baseline");
            c.Name = nameof(c);
            var eq = Op.Eq(c_gold, c);
            c.Reward = eq.As<float>();
            c.Reward.Name = "reward";

            var loss = - c.As<float>() + 0.001f * Op.Norm2(L);
            var grad = Op.Grad(loss); // c isn't differentiable but Reinforce will still backpropagate gradients

            var lr = Op.Scalar<float>("lr");
            var updates = new OrderedDictionary();
            foreach (var W in grad.Keys)
                updates[W] = W - lr * grad[W];

            var dB = (Scalar<float>)loss.Backpropagation.ScalarDerivatives[c.Baseline];
            updates[c.Baseline] = c.Baseline - lr * dB;

            var train_ = Op.Function(new IVar[] { a, c_gold, lr }, c.Reward, updates);
            Train = lr_ => {
                var sample = NextSample();
                return (int)train_(sample.Item1, sample.Item2, lr_);
            };

            Const = Op.Function(a, c);
        }

        public (int, int) NextSample()
        {
            var x = NN.Random.NextInt(N);
            return (x, gold);
        }

        public float TestOn(int n)
        {
            int acc = 0;
            for (int i = 0; i < n; ++i)
            {
                var sample = NextSample();
                if (Const(sample.Item1) == sample.Item2)
                    acc += 1;
            }

            return (float)acc / n;
        }

        public void TrainFor(int epoch, int epochLength)
        {
            for (int e = 0; e < epoch; ++e)
            {
                //float lr = 1f / (1 + e);
                float lr = 0.1f;
                int acc = 0;
                for (int i = 0; i < epochLength; ++i)
                    acc += Train(lr);
                var res = (float)acc / epochLength;
                Trace.WriteLine($"End of epoch {e}, with lr {lr}. Accuracy: {100 * res}%.");
            }
            Trace.WriteLine($"Testing on {epochLength} samples. Accuracy: {100 * TestOn(epochLength)}%.");
        }
    }
}
