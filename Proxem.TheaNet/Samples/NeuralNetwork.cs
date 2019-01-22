using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using Proxem.NumNet;
using T = Proxem.TheaNet.Op;
using Proxem.TheaNet;
using System.IO;
using Proxem.TheaNet.Samples.Optimizer;

namespace Proxem.TheaNet.Samples
{
    public class NeuralNetwork
    {
        // A layer of neurons
        public class Layer
        {
            public Tensor<float>.Shared w;         // Weight vector of this neuron
            public Tensor<float>.Shared b;         // Bias of this neuron
            readonly public Func<Tensor<float>, Tensor<float>> Activation;
            readonly public bool Bias;
            //public System.Random rnd = new System.Random(12345);

            // Weights and biases are "variables" (Expr.Var) initialized between -0.5 and 0.5
            public Layer(int input, int output, Func<Tensor<float>, Tensor<float>> activation, int name = 0, bool bias = true,
                         string savePath = null, string loadPath = null)
            {
                this.Activation = activation;

                if (loadPath != null)
                {
                    using (var reader = new BinaryReader(File.OpenRead(loadPath)))
                    {
                        this.w = T.Shared(Proxem.NumNet.Single.ArrayExtensions.Load(reader), $"w{name}");
                    }
                }
                else
                {
                    this.w = T.Shared(NN.Random.Uniform(-0.5f, 0.5f, input, output), $"w{name}");
                }
                this.Bias = bias;
                if (bias)
                {
                    this.b = T.Shared(NN.Random.Uniform(-0.5f, 0.5f, 1, output), $"b{name}");
                }
                else
                {
                    this.b = T.Shared(NN.Zeros<float>(1, output), $"b{name}");
                }
            }

            public Tensor<float> Forward(Tensor<float> x)
            {
                return this.Activation(T.Dot(x, this.w) + this.b);
                //return this.Activation(T.Dot(x, this.w));
            }
        }

        public static Scalar<float> NegativeBinaryCrossEntropy(Tensor<float> output, Tensor<float> expected)
        {
            return -T.Mean(T.Log(output[T.Range(expected.Shape[0]), expected.As<int>()]));
        }

        /// <summary>A feedforward network of neuron layers</summary>
        public class Network
        {

            public List<Layer> Layers = new List<Layer>();       // The layers forming this network
            //public Tensor<float> eval;
            private Func<Array<float>, Array<float>, float> _trainer;
            private Func<Array<float>, Array<int>> _tester;
            private Func<Array<float>, Array<float>, float> _validator;

            public Network()
            {

            }

            public void Add(Layer layer)
            {
                this.Layers.Add(layer);
            }


            public void Build(GradientDescent Optim, Func<Tensor<float>, Tensor<float>, Scalar<float>> Loss)
            {
                var input = T.Matrix<float>("input");
                var expected = T.Vector<float>("expected"); // scalar si ligne par ligne et vector si en batch
                // learning rate as a placeholder ?

                var outnet = this.Layers.Aggregate((Tensor<float>)input, (x, layer) => layer.Forward(x));

                // Possible fonction différentes 
                //Scalar<float> error = -T.Mean(T.Log(outnet[T.Range(expected.Shape[0]), expected.As<int>()]));
                Scalar<float> error = Loss(outnet, expected);


                // possibilitgé optimizers différents
                //var Optim = new Adam(0.01f);

                var g = T.Grad(error);      // several gradients computed simultaneously // Possible diffenret graident 
                OrderedDictionary updates = new OrderedDictionary();
                var updtg = Optim.UpdateGradient(g);
                foreach (var l in this.Layers)
                {
                    updates[l.w] = l.w - updtg[l.w];
                    if (l.Bias)
                    {
                        updates[l.b] = l.b - updtg[l.b];
                    }
                }

                _trainer = T.Function<Array<float>, Array<float>, float>(
                    input: (input, expected),
                    output: error,
                    updates: updates);

                _tester = T.Function<Array<float>, Array<int>>(
                    input: input,
                    output: T.Argmax(outnet, 1));

                _validator = T.Function<Array<float>, Array<float>, float>(
                    input: (input, expected),
                    output: error);
            }

            public void Train(Array<float> inputs, Array<float> expected, int batchSize, int nbEpochs, float earlyStopThreshold)
            {
                int epoch = 0;
                while (epoch < nbEpochs)
                {
                    int seed = NN.Random.NextInt(10000);
                    NN.Random.Seed(seed);
                    NN.Random.Shuffle<float>(inputs.Values);
                    NN.Random.Seed(seed);
                    NN.Random.Shuffle<float>(expected.Values);


                    double error = 0;
                    int batchNb = inputs.Shape[0] / batchSize;
                    for (int batch = 0; batch < batchNb; batch++)
                    {
                        error += _trainer(inputs[Slicer.Range(batch * batchSize, (batch + 1) * batchSize)],
                                          expected[Slicer.Range(batch * batchSize, (batch + 1) * batchSize)]);
                        if (batch % 1000 == 0)
                        {
                            Console.WriteLine("Batch " + batch + " / " + batchNb + ". error: " + error / (batch + 1));
                        }
                    }
                    error /= batchNb;
                    if (error < earlyStopThreshold)
                    {
                        Console.WriteLine("Computation terminated early -- epoch " + epoch + " / " + nbEpochs + ". Final error: " + error);
                        break;
                    }
                    else
                    {
                        Console.WriteLine("Epoch " + epoch + " / " + nbEpochs + " -- Training Error: " + error);
                        epoch++;
                    }
                }
                //Console.WriteLine("Computation terminated after " + epoch + " epochs. Final error: " + error);
            }

            public int[] Predict(Array<float> inputs)
            {
                var predictions = _tester(inputs).As<int>();
                return predictions.Values;
            }

            public float Validation(Array<float> inputs, Array<float> labels)
            {
                var validError = _validator(inputs, labels);
                Console.WriteLine("Validation Error: " + validError);
                int[] pred = _tester(inputs).As<int>().Values;
                return Score.F1Score(_tester(inputs).As<int>().Values, labels.As<int>().Values, verbose: true);
            }

            public static T[] Shuffle<T>(T[] array)
            {
                for (int i = 0; i < array.Length; i++)
                {

                    var j = NN.Random.NextInt(array.Length);
                    var tmp = array[j];
                    array[j] = array[i];
                    array[i] = tmp;
                }
                return array;
            }
        }
    }
}