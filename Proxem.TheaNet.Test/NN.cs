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

using Proxem.LinearAlgebra.Tensors.Single;
using Proxem.Expressions;
using T = Proxem.Expressions.Op;

namespace Proxem.LinearAlgebra.Expressions.Test
{
    public class TrainingExample : IEnumerable<Tuple<Tensor, Tensor>>
    {
        private readonly List<Tuple<Tensor, Tensor>> content = new List<Tuple<Tensor, Tensor>>();

        public void Add(Tensor input, Tensor target)
        {
            content.Add(Tuple.Create(input, target));
        }

        public void Add(Tuple<Tensor, Tensor> in_out)
        {
            content.Add(in_out);
        }

        public IEnumerator<Tuple<Tensor, Tensor>> GetEnumerator()
        {
            return content.GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return content.GetEnumerator();
        }
    }

    // A layer of neurones
    public class Layer
    {
        public TensorExpr.Var w;         // Weight vector of this neuron
        public TensorExpr.Var b;         // Bias of this neuron
        public Func<TensorExpr, TensorExpr> Activation;

        public TensorExpr Forward(TensorExpr x)
        {
            return Activation(T.Dot(this.w, x) + this.b);
        }
    }

    // A feedforward network of neuron layers
    public class Network
    {
        public static Random rnd = new Random(12345);
        public Layer[] Layers;       // The layers forming this network

        public Network(Layer[] layers)
        {
            this.Layers = layers;
        }

        public static Network WithShape(Func<TensorExpr, TensorExpr> Activation, params int[] layers)
        {

            var Layers = new Layer[layers.Length - 1];
            for (int i = 0; i < layers.Length - 1; i++)
            {
                Layers[i] = new Layer()
                {
                    w = Op.Tensor(Tensor.Random.Uniform(-0.5f, 0.5f, layers[i + 1], layers[i]), "w{0}", i),
                    b = Op.Tensor(Tensor.Zeros(layers[i + 1]), "b{0}", i),
                    Activation = Activation
                };
            }

            return new Network(Layers);
        }

        public TensorExpr Forward(TensorExpr x)
        {
            return Layers.Aggregate(x, (y, l) => l.Forward(y));
        }

        public IEnumerable<float> Backprop(float eta, float epsilon, int timeout, Tuple<Tensor, Tensor>[] ta)
        {
            var input = T.Tensor("input");
            var expected = T.Tensor("expected");
            var output = this.Forward(input);
            var error = 0.5f * T.Norm2(output - expected);              // error is a symbolic expression

            var updates = Op.Updates();
            foreach (var l in this.Layers)
            {
                updates[l.w] = l.w - eta * T.Grad(error, l.w);
                updates[l.b] = l.b - eta * T.Grad(error, l.b);
            }

            var eval = T.Function(input, expected, output: error, updates: updates);

            for (int epoch = 0; epoch < timeout; epoch++)
            {
                float globalError = 0;
                foreach (var t in Shuffle(ta))
                {
                    globalError += eval(t.Item1, t.Item2);
                }
                globalError /= ta.Length;
                yield return globalError;
                if (globalError < epsilon) yield break;
            }
            Console.WriteLine("timeout");
        }

        public static T[] Shuffle<T>(T[] array)
        {
            for (int i = 0; i < array.Length; i++)
            {
                var j = rnd.Next() % array.Length;
                var tmp = array[j];
                array[j] = array[i];
                array[i] = tmp;
            }
            return array;
        }
    }
}
