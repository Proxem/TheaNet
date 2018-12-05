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
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Proxem.TheaNet
{
    public delegate IExpr Derivative(IReadOnlyList<IExpr> inputs, IExpr thiz);

    public static class CustomOp
    {
        public static Scalar<R> Create<R>(string functionName, Func<R> f) =>
            new CustomScalarOp<R>(functionName, f);

        public static Scalar<R> Create<T1, R>(string functionName, Func<T1, R> f, Scalar<T1> x) =>
            new CustomScalarOp<R>(functionName, f, x);

        public static Scalar<R> Create<T1, R>(string functionName, Func<T1, R> f, Func<Scalar<T1>, Scalar<R>, Scalar<R>> df_dx, Scalar<T1> x) =>
            new CustomScalarOp<R>(functionName, f, new[] { x },
                new Derivative[]
                {
                    (args, r) => df_dx((Scalar<T1>)args[0], (Scalar<R>)r)
                }
            );

        public static Scalar<R> Create<T1, T2, R>(string functionName, Func<T1, T2, R> f, Scalar<T1> x, Scalar<T2> y) =>
            new CustomScalarOp<R>(functionName, f, x, y);
    }

    public interface ICustomOp<T>: IExpr<T>
    {
        string CustomFunctionName { get; }
        Delegate Function { get; }
    }

    public class CustomScalarOp<T> : Scalar<T>.NAry, ICustomOp<T>
    {
        public CustomScalarOp(string functionName, Delegate function, params IExpr[] inputs) :
            this(functionName, function, (IReadOnlyList<IExpr>)inputs)
        { }

        public CustomScalarOp(string functionName, Delegate function, IReadOnlyList<IExpr> inputs, IReadOnlyList<Derivative> derivatives = null):
            base("Invoke", inputs.ToArray(), new[] { functionName })
        {
            CustomFunctionName = functionName;
            Function = function;
            _derivatives = derivatives;
        }

        IReadOnlyList<Derivative> _derivatives;
        public string CustomFunctionName { get; }
        public Delegate Function { get; }

        public override void Backward(Scalar<T> delta, Backpropagation bp)
        {
            if(_derivatives == null && Inputs.Count > 0)
                throw new Exception($"The customOp {CustomFunctionName} doesn't support derivation.");

            int n = Inputs.Count;
            for(int i = 0; i < n; ++i)
            {
                if(_derivatives[i] != null)
                {
                    dynamic x = Inputs[i];
                    dynamic d = _derivatives[i](Inputs, delta);
                    bp.PushGradientTo(x, d);
                }
            }
        }

        public override Scalar<T> Clone(IReadOnlyList<IExpr> inputs) => new CustomScalarOp<T>(FunctionName, Function, Inputs);
    }
}
