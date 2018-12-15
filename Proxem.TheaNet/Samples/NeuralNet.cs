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
using Proxem.NumNet;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Samples
{
    public class NeuralNet
    {
        // A layer of neurons
        public class Layer
        {
            public Tensor<float>.Shared w;         // Weight vector of this neuron
            public Tensor<float>.Shared b;         // Bias of this neuron

            public Tensor<float> Forward(Tensor<float> x)
            {
                //return Sigmoid(T.Dot(this.w, x) + this.b);
                return T.Tanh(T.Dot(this.w, x) + this.b);
            }

            public static Tensor<float> Sigmoid(Tensor<float> x)
            {
                return 1 / (1 + T.Exp(-x));
            }
        }

        /// <summary>A feedforward network of neuron layers</summary>
        public class Network
        {
            public static System.Random rnd = new System.Random(12345);

            public Layer[] Layers;       // The layers forming this network

            // Initialize a fully connected feedforward neural network
            // Weights and biases are "variables" (Expr.Var) initialized between -0.5 and 0.5
            public Network(int inputs, params int[] layers)
            {
                this.Layers = new Layer[layers.Length];
                for (int i = 0; i < layers.Length; i++)
                {
                    var w = new float[layers[i]][];
                    var b = new float[layers[i]];
                    for (int j = 0; j < layers[i]; j++)
                    {
                        w[j] = new float[i == 0 ? inputs : layers[i - 1]];
                        for (int k = 0; k < w[j].Length; k++)
                        {
                            w[j][k] = (float)(-0.5 + rnd.NextDouble());
                        }
                        b[j] = (float)(-0.5 + rnd.NextDouble());
                    }
                    this.Layers[i] = new Layer {
                        w = T.Shared(NN.Array<float>(w), $"w{i}"),
                        b = T.Shared(NN.Array<float>(b)/*[_, NewAxis]*/, $"b{i}")
                    };
                }
            }

            public IEnumerable<double> Backprop(float eta, float epsilon, int timeout, Tuple<float[], float[]>[] tf)
            {
                var ta = tf.Select((x, i) => Tuple.Create(
                    NN.Array<float>(x.Item1)/*[_, NewAxis]*/,
                    NN.Array<float>(x.Item2)/*[NewAxis ,_]*/
                )).ToArray();

                var input = T.Vector<float>("input");
                var expected = T.Vector<float>("expected");
                var output = this.Layers.Aggregate((Tensor<float>)input, (x, layer) => layer.Forward(x));
                var error = 0.5f * T.Norm2(output - expected);              // error is a symbolic expression

                var updates = new OrderedDictionary();
                foreach (var l in this.Layers)
                {
                    var g = T.Grad(error);      // several gradients computed simultaneously
                    updates[l.w] = l.w - eta * g[l.w];
                    updates[l.b] = l.b - eta * g[l.b];

                    //updates[l.w] = l.w - eta * T.Grad(error, l.w);
                    //updates[l.b] = l.b - eta * T.Grad(error, l.b);
                }

                var eval = T.Function(input: (input, expected), output: error, updates: updates);

                for (int epoch = 0; epoch < timeout; epoch++)
                {
                    double globalError = 0;
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
}
