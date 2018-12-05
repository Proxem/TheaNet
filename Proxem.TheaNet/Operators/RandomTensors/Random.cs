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

namespace Proxem.TheaNet.Operators
{
    using Dim = Scalar<int>;

    public sealed class RandomFactory
    {
        public Tensor<T> Uniform<T>(Scalar<T> min, Scalar<T> max, params Dim[] shape)
        {
            return new Uniform<T>(min, max, shape);
        }

        public Tensor<T> Uniform<T>(Scalar<T> min, Scalar<T> max, XList<Scalar<int>, int> shape)
        {
            return new Uniform<T>(min, max, shape);
        }

        public Tensor<T> Normal<T>(Scalar<T> mean, Scalar<T> std, params Dim[] shape) => new Normal<T>(mean, std, shape);
    }

    public class Uniform<T> : Tensor<T>.NAry
    {
        public Uniform(Scalar<T> min, Scalar<T> max, XList<Scalar<int>, int> shape) : base("Uniform", min, max, shape) { }

        public Scalar<T> Min => (Scalar<T>)Inputs[0];
        public Scalar<T> Max => (Scalar<T>)Inputs[1];
        public override sealed Dim[] Shape => (XList<Scalar<int>, int>)Inputs[2];

        public override void Backward(Tensor<T> delta, Backpropagation bp)
        {
            throw new NotImplementedException();
        }

        public override NAry Clone(IReadOnlyList<IExpr> inputs)
        {
            return new Uniform<T>((Scalar<T>)inputs[0], (Scalar<T>)inputs[1], (XList<Scalar<int>, int>)inputs[2]);
        }
    }

    public class Normal<T> : Tensor<T>.NAry
    {
        public Normal(Scalar<T> mean, Scalar<T> std, XList<Scalar<int>, int> shape) : base("Normal", mean, std, shape) { }

        public Scalar<T> Mean => (Scalar<T>)Inputs[0];
        public Scalar<T> Variance => (Scalar<T>)Inputs[1];

        public override sealed Dim[] Shape => (XList<Scalar<int>, int>)Inputs[2];

        public override void Backward(Tensor<T> delta, Backpropagation bp) { }

        public override NAry Clone(IReadOnlyList<IExpr> inputs)
        {
            return new Normal<T>((Scalar<T>)inputs[0], (Scalar<T>)inputs[1], (XList<Scalar<int>, int>)inputs[2]);
        }
    }
}
