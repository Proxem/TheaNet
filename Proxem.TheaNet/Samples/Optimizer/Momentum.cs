using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using Proxem.NumNet;
using T = Proxem.TheaNet.Op;
using Proxem.TheaNet;

namespace Proxem.TheaNet.Samples.Optimizer
{
    class Momentum : GradientDescent
    {
        public float MomentumTerm { get; set; }

        public Dictionary<Tensor<float>.Symbol, Tensor<float>> Update { get; private set; }

        public Momentum(float learningRate, float momentum = 0.9f) : base(learningRate: learningRate)
        {
            
            MomentumTerm = momentum;
            Update = new Dictionary<Tensor<float>.Symbol, Tensor<float>>();
        }

        public override Dictionary<Tensor<float>.Symbol, Tensor<float>> UpdateGradient(Dictionary<Tensor<float>.Symbol, Tensor<float>> gradient)
        {
            foreach (Tensor<float>.Symbol param in gradient.Keys)
            {
                if (Update.ContainsKey(param))
                {
                    Update[param] = MomentumTerm * Update[param] + LearningRate * gradient[param];
                }
                else
                {
                    Update[param] = LearningRate * gradient[param];
                }
            }
            return Update;
        }
    }
}
