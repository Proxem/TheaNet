using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using Proxem.NumNet;
using T = Proxem.TheaNet.Op;
using Proxem.TheaNet;

namespace Proxem.TheaNet.Samples.Optimizer
{
    class Adam : GradientDescent
    {
        public float Beta1 { get; set; }

        public float Beta2 { get; set; }

        public float Epsilon { get; set; }

        public Dictionary<Tensor<float>.Symbol, Tensor<float>> MomentOne { get; private set; }

        public Dictionary<Tensor<float>.Symbol, Tensor<float>> MomentTwo { get; private set; }

        public Adam(float learningRate = 0.1f,
                    float beta1 = 0.9f,
                    float beta2 = 0.999f,
                    float epsilon = 1e-8f) : base(learningRate)
        {
            Beta1 = beta1;
            Beta2 = beta2;
            Epsilon = epsilon;
            MomentOne = new Dictionary<Tensor<float>.Symbol, Tensor<float>>();
            MomentTwo = new Dictionary<Tensor<float>.Symbol, Tensor<float>>();
        }

        public override Dictionary<Tensor<float>.Symbol, Tensor<float>> UpdateGradient(Dictionary<Tensor<float>.Symbol, Tensor<float>> gradient)
        {

            Dictionary<Tensor<float>.Symbol, Tensor<float>> uptdGrad = new Dictionary<Tensor<float>.Symbol, Tensor<float>>();

            foreach (Tensor<float>.Symbol param in gradient.Keys)
            {
                if (MomentOne.ContainsKey(param))
                {
                    MomentOne[param] = Beta1 * MomentOne[param] + (1 - Beta1) * gradient[param];
                }
                else
                {
                    MomentOne[param] = (1 - Beta1) * gradient[param];
                }

                if (MomentTwo.ContainsKey(param))
                {
                    MomentTwo[param] = Beta2 * MomentTwo[param] + (1 - Beta2) * T.Pow(gradient[param], 2);
                }
                else
                {
                    MomentTwo[param] = (1 - Beta2) * T.Pow(gradient[param], 2);
                }

                var BiasCOrrectedMomentOne = MomentOne[param] / (1 - Beta1);
                var BiasCOrrectedMomentTwo = MomentTwo[param] / (1 - Beta2);

                uptdGrad[param] = LearningRate / (T.Sqrt(BiasCOrrectedMomentTwo) + Epsilon) * BiasCOrrectedMomentOne;
            }
            return uptdGrad;
        }
    }
}
