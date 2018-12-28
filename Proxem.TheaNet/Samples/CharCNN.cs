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
using System.Collections.Specialized;
using System.Linq;

using Proxem.NumNet;
using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Samples
{
    public class CharCNN
    {
        // lexicon
        Tensor<float>.Shared L;
        // conv kernel
        Tensor<float>.Shared K;

        // prediction layer
        Tensor<float>.Shared S;
        Tensor<float>.Shared Sb;

        public readonly Tensor<float>.Shared[] @params;
        public readonly Scalar<float> Loss;
        public readonly Tensor<int>.Var chars;
        public readonly Scalar<int>.Var gold;

        public Func<Array<int>, int> _classify;
        public Func<string, int> classify;

        public Func<Array<int>, int, float, float> _train;
        public Func<string, int, float, float> train;

        public CharCNN(int embeddingSize, int kernelSize, int hiddenSize, int nclasses = 2)
        {
            int vocabSize = ALPHA_LENGTH + 1;

            L = T.Shared(NN.Random.Uniform(-1f, 1f, vocabSize, embeddingSize), nameof(L));

            var scaleK = 0.1f;
            K = T.Shared(NN.Random.Uniform(-scaleK, scaleK, kernelSize, embeddingSize, hiddenSize), nameof(K));

            var scaleS = 0.1f;
            S = T.Shared(NN.Random.Uniform(-scaleS, scaleS, hiddenSize, nclasses), nameof(S));
            Sb = T.Shared(NN.Zeros(nclasses), nameof(Sb));

            @params = new[] { L, K, S, Sb };

            chars = T.Vector<int>("chars");
            var word = L[chars]; word.Name = nameof(word);

            var x = T.ConvolveSentence(word, K); x.Name = nameof(x);
            var x_pooled = T.ReLu(T.Max(x, 0)); x_pooled.Name = nameof(x_pooled);

            var pred = T.Softmax(T.Dot(x_pooled, S) + Sb);
            _classify = T.Function(input: chars, output: T.Argmax(pred));
            classify = w => _classify(Convert(w));

            gold = T.Scalar<int>("gold");
            var loss = -T.Log(pred.Item[gold]);

            var reg = 0.001f;
            foreach (var W in new[] { K, S })
                loss += reg * T.Norm2(W);
            loss.Name = nameof(loss);

            Loss = loss;
            var grad = T.Grad(Loss);

            var lr = T.Scalar<float>("lr");
            var updates = new OrderedDictionary();
            foreach (var W in @params)
                updates[W] = W - lr * grad[W];

            _train = T.Function(input: (chars, gold, lr), output: Loss, updates: updates);
            train = (w, g, l) => _train(Convert(w), g, l);
        }

        public float DoOneEpoch(float lr = 0.1f)
        {
            float err = 0;
            int n = Math.Min(FrenchVerb1Train.Length, FrenchVerb2Train.Length);
            for (int i = 0; i < n; ++i)
            {
                err += train(FrenchVerb1Train[i], 0, 2 * lr);
                err += train(FrenchVerb2Train[i], 1, lr);
            }
            err /= 2 * n;
            return err;
        }

        public float TrainAccuracy() =>
            Accuracy(FrenchVerb1Train, FrenchVerb2Train);

        public float TestAccuracy() =>
            Accuracy(FrenchVerb1Test, FrenchVerb2Test);

        public float Accuracy(string[] class0, string[] class1)
        {
            int right = 0;
            for (int i = 0; i < class0.Length; ++i)
                if (classify(class0[i]) == 0)
                    ++right;

            for (int i = 0; i < class1.Length; ++i)
                if (classify(class1[i]) == 1)
                    ++right;

            return ((float)right) / (class0.Length + class1.Length);
        }

        const int MIN_CHAR = 'a';
        const int ALPHA_LENGTH = 26;

        Array<int> Convert(string word)
            => NN.Array( word
                .Select(c => c - MIN_CHAR)
                .Select(i => i < 0 || i > ALPHA_LENGTH ? ALPHA_LENGTH : i)
                .ToArray()
            );

        static string[] FrenchVerb1Train = new string[]
        {
            "aimer", "commencer", "lever", "amener", "transférer", "inquiéter", "manger", "assiéger",
            "appeler", "acheter", "créer", "oublier", "payer", "noyer", "ennuyer"
        };

        static string[] FrenchVerb1Test = new string[]
        {
            "chanter", "constituer", "solutionner", "rentrer", "rentrer", "délaisser", "arriver", "exercer"
        };

        static string[] FrenchVerb2Train = new string[]
        {
            "finir", "choisir", "jaunir", "atterir", "rougir", "grandir", "élargir", "obéir",
            "réussir", "maigrir", "ravir", "moisir", "réagir", "mugir", "épanouir"
        };

        static string[] FrenchVerb2Test = new string[]
        {
            "établir", "rôtir", "accomplir", "bâtir", "divertir", "réjouir", "réunir", "remplir"
        };
    }
}
