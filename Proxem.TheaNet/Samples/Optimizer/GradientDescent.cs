using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using Proxem.NumNet;
using T = Proxem.TheaNet.Op;
using Proxem.TheaNet;

namespace Proxem.TheaNet.Samples.Optimizer
{
    public class GradientDescent
    {
        //public float LearningRate;
        public float LearningRate { get; private set; }

        public GradientDescent(float learningRate)
        {
            LearningRate = learningRate;
        }

        public virtual Dictionary<Tensor<float>.Symbol, Tensor<float>> UpdateGradient(Dictionary<Tensor<float>.Symbol, Tensor<float>> gradient)
        {

            Dictionary<Tensor<float>.Symbol, Tensor<float>> uptdGrad = new Dictionary<Tensor<float>.Symbol, Tensor<float>>();

            foreach (Tensor<float>.Symbol param in gradient.Keys)
            {
                uptdGrad[param] = LearningRate * gradient[param];
            }
            return uptdGrad;
        }
    }
}
