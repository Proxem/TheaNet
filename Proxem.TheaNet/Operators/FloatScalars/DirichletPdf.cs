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

namespace Proxem.TheaNet.Operators.FloatScalars
{
    public class DirichletPdf : Scalar<float>.NAry
    {
        public DirichletPdf(Scalar<float> alpha, Tensor<float> x) : base("DirichletPdf", alpha, x) { }

        public Scalar<float> alpha => (Scalar<float>)this.Inputs[0];
        public Tensor<float> x => (Tensor<float>)this.Inputs[1];

        public override void Backward(Scalar<float> delta, Backpropagation bp)
        {
            bp.PushGradientTo(x, (alpha - 1) * (this * delta) / x);

            // TODO: learn alpha
            if(!(alpha is Const))
                throw new NotImplementedException();
        }

        public override Scalar<float> Clone(IReadOnlyList<IExpr> inputs) =>
            new DirichletPdf((Scalar<float>)inputs[0], (Tensor<float>)inputs[1]);
    }
}
