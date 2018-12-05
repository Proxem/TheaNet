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
using System.Globalization;
using System.Linq;

namespace Proxem.TheaNet.Binding
{
    public class InlineCodeGenerator: IProcessor
    {
        public string Result;
        private Tuple<IVar, IExpr>[] bindings;

        public static string GetCode(IExpr target)
        {
            var generator = new InlineCodeGenerator(null);
            return generator.GetCode(target, int.MaxValue);
        }

        public static string Process(IExpr expr, Tuple<IVar, IExpr>[] bindings = null)
        {
            var processor = new InlineCodeGenerator(bindings);
            expr.Process(processor);
            return processor.Result;
        }

        public InlineCodeGenerator(Tuple<IVar, IExpr>[] bindings)
        {
            this.bindings = bindings;
        }

        public string GetCode(IExpr target, int precedence)
        {
            if (bindings != null)
            {
                for (int i = 0; i < bindings.Length; i++)
                {
                    if (bindings[i].Item1 == target)
                    {
                        target = bindings[i].Item2;
                        break;
                    }
                }
            }
            if (target.Name != null) return target.Name;
            target.Process(this);
            var code = this.Result;
            return precedence < Compiler.Precedences(target) ? "(" + code + ")" : code;
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
            this.Result = $"[{string.Join(", ", target.Inputs.Select(a => this.GetCode(a, int.MaxValue)))}]";
        }

        public void ObsoleteProcessArray<T, U>(IExpr<T> target, IEnumerable<IExpr<U>> args)
        {
            this.Result = $"[{string.Join(", ", args.Select(a => this.GetCode(a, int.MaxValue)))}]";
        }

        public void ProcessFunctionCall<T>(IExpr<T> target, string name, params object[] extras)
        {
            var args = target.Inputs;
            switch (name)
            {
                case "[]":
                    if (extras != null)
                    {
                        this.Result = $"{ToString(target, args[0], 0)}{string.Join(", ", extras.Select(arg => ToString(target, arg)))}";
                    }
                    else
                    {
                        this.Result = $"{ToString(target, args[0], 0)}{ToString(target, args[1])}";
                    }
                    break;
                case "Shape":
                    this.Result = $"{ToString(target, args[0], 0)}.{name}[{ToString(target, args[1], 0)}]";
                    break;
                case "Item":
                    this.Result = $"{ToString(target, args[0], 0)}.{name}{ToString(target, args[1], 0)}";
                    break;
                case "CastScalar":
                case "CastTensor":
                    this.Result = "(" + typeof(T).GetName() + ")" + this.GetCode(args[0], 0);
                    break;
                case "TupleItem":
                    this.Result = $"{this.GetCode(args[0], 0)}.Item{extras[0]}";
                    break;
                case "Add":
                    this.Result = $"{ToString(target, args[0], 3)} + {ToString(target, args[1], 3)}";
                    break;
                case "Div":
                    this.Result = $"{ToString(target, args[0], 2)} / {ToString(target, args[1], 2)}";
                    break;
                case "Ge":
                    this.Result = $"{ToString(target, args[0], 5)} >= {ToString(target, args[1], 5)})";
                    break;
                case "Gt":
                    this.Result = $"{ToString(target, args[0], 5)} > {ToString(target, extras == null ? args[1] : extras[0], 5)})";
                    break;
                case "Mean":
                    this.Result = $"{ToString(target, args[0], 0)}.Mean()";
                    break;
                case "Mod":
                    this.Result = $"{ToString(target, args[0], 2)} % {ToString(target, args[1], 2)}";
                    break;
                case "Mul":
                    this.Result = $"{ToString(target, args[0], 2)} * {ToString(target, args[1], 2)}";
                    break;
                case "Neg":
                    this.Result = $"-{ToString(target, args[0], 1)}";
                    break;
                case "Neq":
                    this.Result = $"{ToString(target, args[0], 6)} != {ToString(target, args[1], 6)}";
                    break;
                case "Reshape":
                    this.Result = $"{ToString(target, args[0], 0)}.Reshape({ToString(target, args[1])})";
                    break;
                case "Size":
                    this.Result = $"{ToString(target, args[0], 0)}.Size";
                    break;
                case "Sub":
                    this.Result = $"{ToString(target, args[0], 3)} - {ToString(target, args[1], 3)}";
                    break;
                case "Sum":
                    this.Result = $"{ToString(target, args[0], 0)}.Sum({string.Join(", ", args.Skip(1).Select(arg => ToString(target, arg)))})";
                    break;
                case "Array":
                    this.Result = $"[{string.Join(", ", args.Select(arg => ToString(target, arg)))}]";
                    break;
                case "Print":
                    var format = (string)extras[0];
                    if (format != null)
                        this.Result = $"Print(\"{format.Replace("{", "{{").Replace("}", "}}")}\", {this.GetCode(args[0], 0)})";
                    else
                        this.Result = $"Print({this.GetCode(args[0], 0)})";
                    break;
                case "Invoke":
                    var fn = ((ICustomOp<T>)target).CustomFunctionName;
                    this.Result = $"{name}(\"{fn}\"";
                    if(args.Count > 0)
                        Result += ", " + string.Join(", ", args.Select(arg => ToString(target, arg)));
                    Result += ")";
                    break;
                default:
                    this.Result = name + "(" + string.Join(", ", args.Select(arg => ToString(target, arg))) + ")";
                    break;
            }
        }

        private void GenerateLambda(Lambda lambda, int precedence)
        {
            var lvars = string.Join(", ", lambda.Vars.Select(arg => this.GetCode(arg, int.MaxValue)));
            if (lambda.Vars.Length > 1) lvars = $"({lvars})";

            var content = this.GetCode(lambda.Expr, int.MaxValue);

            this.Result = lvars + " => " + content;
        }

        public void ProcessTuple(ITuple target, params IExpr[] args)
        {
            this.Result = $"({string.Join(", ", args.Select(arg => this.GetCode(arg, int.MaxValue)))})";
        }

        public void ProcessFor<T>(Tensor<T>.For target)
        {
            this.Result = $"For{{({string.Join(", ", target.Loop.Variables)}) => {target.Expression.Comment}}}";
            //result = result.Replace("{", "{{").Replace("}", "}}");
        }

        private string ToString(IExpr expr)
        {
            return Convert.ToString(expr, CultureInfo.InvariantCulture);
        }

        private string ToString(XSlice slice)
        {
            if (slice.IsSingleton)
            {
                return this.ToString((IExpr)slice.Start);
            }
            if (slice.Stop.Equals(Numeric<int>.FromInt(int.MaxValue)))
            {
                if (slice.Start.Equals(Numeric<int>.Zero))
                {
                    if (slice.Step.Equals(Numeric<int>.One)) return "Slicer._";
                    return "Slicer.Step(" + this.ToString((IExpr)slice.Step) + ")";
                }
                var start = this.ToString((IExpr)slice.Start);
                if (slice.Step.Equals(Numeric<int>.One)) return "Slicer.From(" + start + ")";
                var step = this.ToString((IExpr)slice.Step);
                return "Slicer.From(" + start + ", " + step + ")";
            }
            if (slice.Step is Scalar<int>.Const conststep && conststep.Value < 0 && slice.Start.Equals(Numeric<int>.FromInt(-1)) && slice.Stop.Equals(Numeric<int>.FromInt(int.MinValue))) return "Step(" + conststep.Value + ")";
            var stop = this.ToString((IExpr)slice.Stop);
            if (slice.Start.Equals(Numeric<int>.Zero))
            {
                if (slice.Step.Equals(Numeric<int>.One)) return "Slicer.Until(" + stop + ")";
                return "Slicer.Until(" + stop + ", " + this.ToString((IExpr)slice.Step) + ")";
            }
            {
                var start = this.ToString((IExpr)slice.Start);
                if (slice.Step.Equals(Numeric<int>.One)) return "Slicer.Range(" + start + ", " + stop + ")";
                var step = this.ToString((IExpr)slice.Step);
                return "Slicer.Range(" + start + ", " + stop + ", " + step + ")";
            }
        }

        private string ToString<T>(IExpr<T> target, object arg, int precedence = int.MaxValue)
        {
            switch (arg)
            {
                case IExpr expr:
                    return this.GetCode(expr, precedence);
                case IEnumerable<IExpr<int>> enumi:
                    ObsoleteProcessArray(target, enumi);
                    return this.Result;
                case IEnumerable<IExpr<float>> enumf:
                    ObsoleteProcessArray(target, enumf);
                    return this.Result;
                case IEnumerable<IExpr<NumNet.Array<int>>> enumai:
                    ObsoleteProcessArray(target, enumai);
                    return this.Result;
                case IEnumerable<IExpr<NumNet.Array<float>>> enumaf:
                    ObsoleteProcessArray(target, enumaf);
                    return this.Result;
                case IEnumerable<XSlice> enumx:
                    ObsoleteProcessArray(target, enumx);
                    return this.Result;
                //case Tensor<int>[] ti:
                //    ObsoleteProcessArray(target, ti);
                //    return this.Result;
                case NamedObject no:
                    return no.Name + ": " + ToString(target, no.Object, precedence);
                case int i:
                    return arg.ToString();
                case int[] ai:
                    return $"[{string.Join(", ", ai)}]";
                case float f:
                    return arg.ToString();
                case bool b:
                    return arg.ToString().ToLower();
                case string s:
                    return s;
                case Lambda l:
                    GenerateLambda(l, precedence);
                    return this.Result;
                case Enum e:
                    return e.ToString();
                default:
                    throw new NotImplementedException(arg.GetType().GetName());
            }
        }

        public void ProcessSlice(XSlice target)
        {
            this.Result = ToString(target);
        }

        public void ProcessElementwise<T>(Tensor<T>.Elementwise target)
        {
            var mapping = target.Vars.Zip(target.Inputs, Tuple.Create<IVar, IExpr>);
            var bindings = this.bindings != null ? this.bindings.Concat(mapping).ToArray() : mapping.ToArray();
            this.Result = InlineCodeGenerator.Process(target.Abstraction, bindings);
        }
    }
}
