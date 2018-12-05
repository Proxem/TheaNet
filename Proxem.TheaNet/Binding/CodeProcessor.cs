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
using System.Diagnostics;
using System.Globalization;
using System.Linq;

namespace Proxem.TheaNet.Binding
{
    /// <summary>
    /// Converts a TheaNet expression in a C# expression
    /// </summary>
    public class CodeProcessor : IProcessor
    {
        private Compiler Compiler;
        private string Result;
        private Tuple<IVar, IExpr>[] Bindings;

        public static string Process(Compiler compiler, IExpr target, params Tuple<IVar, IExpr>[] bindings)
        {
            var processor = new CodeProcessor(compiler, bindings);
            target.Process(processor);
            return processor.Result;
        }

        private CodeProcessor(Compiler compiler, Tuple<IVar, IExpr>[] bindings)
        {
            this.Compiler = compiler;
            this.Bindings = bindings;
        }

        private string GetCode(IExpr target)
        {
            if (this.Compiler.Scope.Contains(target))
                return this.Compiler.Scope.GetVar(target);

            for (int i = 0; i < this.Bindings.Length; i++)
            {
                if (this.Bindings[i].Item1 == target)
                {
                    target = this.Bindings[i].Item2;
                    break;
                }
            }
            target.Process(this);
            return this.Result;
        }

        private string GetCode(IExpr target, int precedence = int.MaxValue)
        {
            var code = GetCode(target);
            return precedence < Compiler.Precedences(target) ? "(" + code + ")" : code;
        }

        private string GetCode(object arg)
        {
            // <remove>
            //if (arg is IExpr) return this.GetCode((IExpr)arg);
            if (arg is IEnumerable<IExpr<int>>) { ObsoleteProcessArray((IEnumerable<IExpr<int>>)arg); return this.Result; }
            if (arg is IEnumerable<IExpr<float>>) { ObsoleteProcessArray((IEnumerable<IExpr<float>>)arg); return this.Result; }
            if (arg is IEnumerable<IExpr<NumNet.Array<int>>>) { ObsoleteProcessArray((IEnumerable<IExpr<NumNet.Array<int>>>)arg); return this.Result; }
            if (arg is IEnumerable<IExpr<NumNet.Array<float>>>) { ObsoleteProcessArray((IEnumerable<IExpr<NumNet.Array<float>>>)arg); return this.Result; }
            if (arg is IEnumerable<XSlice>) { ObsoleteProcessArray((IEnumerable<XSlice>)arg); return this.Result; }
            //if (arg is Tensor<int>[]) { ObsoleteProcessArray((Tensor<int>[])arg); return this.Result; }
            // compiler chokes on preceding line (Visual C# 2017 RC)
            var arg1 = arg as Tensor<int>[];
            if (arg1 != null) { ObsoleteProcessArray(arg1); return this.Result; }
            // </remove>

            if (arg is NamedObject) return ((NamedObject)arg).Name + ": " + GetCode(((NamedObject)arg).Object);
            if (arg is int) return arg.ToString();
            if (arg is int[]) return $"new int[] {{{string.Join(", ", (int[])arg)}}}";
            if (arg is float) return ((float)arg).ToString(CultureInfo.InvariantCulture) + "f";
            if (arg is bool) return arg.ToString().ToLower();
            if (arg is string) return $"\"{arg}\"";
            if (arg is Enum) return ((int)arg).ToString();
            if (arg is Lambda)
            {
                var lambda = (Lambda)arg;
                var lvars = string.Join(", ", lambda.Vars.Select(var => GetCode(var)));
                if (lambda.Vars.Length > 1) lvars = $"({lvars})";

                // creates a scope for the lambda
                using (var scope = Compiler.CreateScope())
                {
                    // inside this scope the arguments of the lambda exist
                    foreach (var @var in lambda.Vars)
                        Compiler.Scope.Declare(@var, Compiler);

                    return lvars + " => " + this.GetCode(lambda.Expr);
                }
            }
            throw new NotImplementedException(arg.GetType().GetName());
        }

        public void ProcessLiteral<T>(IExpr<T> target, T value)
        {
            this.Result = Numeric.GetLiteral<T>(value);
        }

        public void ProcessVar<T>(IVar<T> target)
        {
            this.Result = target.Name;
        }

        public void ProcessShared<T>(IShared<T> target)
        {
            this.Result = target.Name;
        }

        void IProcessor.ProcessList<T, U>(XList<T, U> target)
        {
            this.Result = $"new {typeof(U).GetName()}[] {{{string.Join(", ", target.Inputs.Select(a => this.GetCode(a)))}}}";
        }

        public void ObsoleteProcessArray<U>(IEnumerable<IExpr<U>> args)
        {
            this.Result = $"new {typeof(U).GetName()}[] {{{string.Join(", ", args.Select(a => this.GetCode(a)))}}}";
        }

        public void ProcessFunctionCall<T>(IExpr<T> target, string name, params object[] extras)
        {
            var args = target.Inputs;
            if (extras != null && extras.Any(x => x is IExpr || x is IEnumerable<IExpr>))
                throw new ArgumentException($"{target} uses hidden !: [{string.Join(", ", extras)}]");
            switch (name)
            {
                case "[]":
                    this.Result = $"{GetCode(args[0], 0)}[{GetCode(args[1])}]";
                    break;
                case "Shape":
                case "Item":
                    this.Result = $"{GetCode(args[0], 0)}.{name}[{GetCode(args[1], 0)}]";
                    break;
                case "CastScalar":
                    this.Result = "(" + typeof(T).GetName() + ")" + GetCode(args[0], 0);
                    break;
                case "CastTensor":
                    // T is NumNet.Array<U>
                    // typeof(T).GenericTypeArguments[0] is U
                    this.Result = this.GetCode(args[0], 0) + ".As<" + typeof(T).GenericTypeArguments[0].GetName() + ">()";
                    break;
                case "TupleItem":
                    this.Result = $"{this.GetCode(args[0], 0)}.Item{extras[0]}";
                    break;
                case "Add":
                    this.Result = $"{GetCode(args[0], 3)} + {GetCode(args[1], 3)}";
                    break;
                case "Div":
                    this.Result = $"{GetCode(args[0], 2)} / {GetCode(args[1], 2)}";
                    break;
                case "Mod":
                    this.Result = $"{GetCode(args[0], 2)} % {GetCode(args[1], 2)}";
                    break;
                case "Mul":
                    this.Result = $"{GetCode(args[0], 2)} * {GetCode(args[1], 2)}";
                    break;
                case "Neg":
                    this.Result = $"-{GetCode(args[0], 1)}";
                    break;
                case "Neq":
                    this.Result = $"{GetCode(args[0], 6)}.Neq({GetCode(args[1], 6)})";
                    break;
                case "Reshape":
                    this.Result = $"{GetCode(args[0], 0)}.Reshape({GetCode(args[1])})";
                    break;
                case "Sign":
                    this.Result = $"({GetCode(args[0])}) > 0 ? 1 : 0";
                    break;
                case "Size":
                    this.Result = $"{GetCode(args[0], 0)}.Size";
                    break;
                case "Sub":
                    this.Result = $"{GetCode(args[0], 3)} - {GetCode(args[1], 3)}";
                    break;
                case "Array":
                    //this.Result = $"new {typeof(T).GetName()} {{{string.Join(", ", args.Select(a => this.GetCode(a, bindings)))}}}";
                    //break;
                    throw new NotImplementedException();
                case "Print":
                    var format = extras[0];
                    if (format != null)
                        this.Result = $"Print(\"{format}\", {GetCode(args[0])})";
                    else
                        this.Result = $"Print({GetCode(args[0])})";
                    break;
                case "IndexWith":
                    this.Result = $"{GetCode(args[0], 0)}.IndexWith({GetCode(args[1])})";
                    break;
                case "Invoke":
                    this.Result = _processInvoke((ICustomOp<T>)target);
                    break;
                case "Broadcast":
                    //Trace.WriteLine($"Broadcasted {target, args[0]} is copied along axes {string.Join(", ", broadcast)}: {this.x.Shape.Format()}->{target, args[1]} at line {Compiler.CurrentLine}.", "Warning, Time efficiency:");
                    Trace.WriteLine($"Broadcasted {args[0]} is copied along axes ...: -> [{string.Join(", ", args.Skip(1))}] at line {Compiler.CurrentLine}.", "Warning, Time efficiency:");
                    goto default;
                case "Const":
                    if ((args[0] as Scalar<int>)?.IsZero ?? false)
                        Trace.WriteLine($"Warning: creating Zeros");
                    goto default;
                case "EinsteinSum":
                    Trace.TraceWarning($"Couldn't simplify Einstein sum: '{args[2]}'");
                    goto default;
                default:
                    if (extras != null)
                    {
                        this.Result = $"{name}({string.Join(", ", args.Select(arg => GetCode(arg)))}, {string.Join(", ", extras.Select(extra => GetCode(extra)))})";
                    }
                    else
                    {
                        this.Result = $"{name}({string.Join(", ", args.Select(arg => GetCode(arg)))})";
                    }
                    break;
            }

            if (target is ITensor tensor && name != "Reshape")
            {
                var buff = Compiler.GetBuffer(tensor);
                int n = Result.Length;
                if (buff != null && Result[n-1] == ')')
                {
                    // handle the case where the called function doesn't have arguments
                    if(Result[n-2] == '(')
                        Result = $"{Result.Substring(0, n - 1)}result: {Compiler.GetBufferName(buff)})";
                    else
                        Result = $"{Result.Substring(0, n - 1)}, result: {Compiler.GetBufferName(buff)})";
                }
            }
        }

        public void ProcessTuple(ITuple target, params IExpr[] args)
        {
            // TODO: we don't need ProcessTuple, the point of Tuple is to describe function returning tuple
            this.Result = $"Tuple.Create({string.Join(", ", args.Select(arg => this.GetCode(arg)))})";
        }

        public void ProcessFor<T>(Tensor<T>.For target)
        {
            this.Result = $"For{{({string.Join(", ", target.Loop.Variables)}) => {target.Expression.Comment}}}";
        }

        public void ProcessSlice(XSlice slice)
        {
            if (slice.IsSingleton)
            {
                this.Result = GetCode((IExpr)slice.Start);
                return;
            }
            if (slice.Stop.Equals(Numeric<int>.FromInt(int.MaxValue)))
            {
                if (slice.Start.Equals(Numeric<int>.Zero))
                {
                    if (slice.Step.Equals(Numeric<int>.One)) this.Result = "Slicer._";
                    else this.Result = "Slicer.Step(" + GetCode((IExpr)slice.Step) + ")";
                    return;
                }
                var start = GetCode((IExpr)slice.Start);
                if (slice.Step.Equals(Numeric<int>.One)) this.Result = "Slicer.From(" + start + ")";
                else
                {
                    var step = GetCode((IExpr)slice.Step);
                    this.Result = "Slicer.From(" + start + ", " + step + ")";
                }
                return;
            }
            if (slice.Step is Scalar<int>.Const conststep
                && conststep.Value < 0 && slice.Start.Equals(Numeric<int>.FromInt(-1)) && slice.Stop.Equals(Numeric<int>.FromInt(int.MinValue)))
            {
                this.Result = "Slicer.Step(" + conststep.Value + ")";
                return;
            }
            var stop = GetCode((IExpr)slice.Stop);
            if (slice.Start.Equals(Numeric<int>.Zero))
            {
                if (slice.Step.Equals(Numeric<int>.One)) this.Result = "Slicer.Until(" + stop + ")";
                else this.Result = "Slicer.Until(" + stop + ", " + GetCode((IExpr)slice.Step) + ")";
            }
            else
            {
                var start = GetCode((IExpr)slice.Start);
                if (slice.Step.Equals(Numeric<int>.One)) this.Result = "Slicer.Range(" + start + ", " + stop + ")";
                else
                {
                    var step = GetCode((IExpr)slice.Step);
                    this.Result = "Slicer.Range(" + start + ", " + stop + ", " + step + ")";
                }
            }
        }

        public void ProcessElementwise<T>(Tensor<T>.Elementwise target)
        {
            // the elementwise is optimized by the CodeGenerator so it's not generally called.
            // This only server as default implementation
            var mapping = target.Vars.Zip(target.Inputs, Tuple.Create<IVar, IExpr>);
            var bindings = this.Bindings.Concat(mapping).ToArray();
            Result = CodeProcessor.Process(Compiler, target.Abstraction, bindings);
        }

        private string _processInvoke<T>(ICustomOp<T> target)
        {
            Compiler.CustomFunctions[target.CustomFunctionName] = target.Function;

            var method = target.Function.Method;
            var returnType = method.ReturnType.Name;
            var args = string.Join(", ", target.Inputs.Select(arg => GetCode(arg)));

            return $"{target.FunctionName}<{returnType}>(\"{target.CustomFunctionName}\", {args})";
        }
    }
}