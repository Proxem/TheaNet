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

namespace Proxem.TheaNet.Operators.Scalars
{
    /// <summary>Returns the position of the bigger element.</summary>
    public class Argmax<Type> : Scalar<int>.NAry
    {
        public Argmax(Tensor<Type> x) : base("Argmax", x)
        {
        }

        public Tensor<Type> x => (Tensor<Type>)this.Inputs.First();

        public override Scalar<int> Clone(IReadOnlyList<IExpr> inputs) => new Argmax<Type>((Tensor<Type>)inputs[0]);

        public override void Backward(Scalar<int> delta, Backpropagation bp)
        {
            throw new NotImplementedException();
        }
    }
}
