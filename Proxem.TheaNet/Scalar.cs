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
using Proxem.NumNet;
using Proxem.TheaNet;
using Proxem.TheaNet.Binding;
using Proxem.TheaNet.Operators.Scalars;

namespace Proxem.TheaNet
{
    public abstract class Scalar<Type> : Expr, IExpr<Type>, IScalar<Type>, IDifferentiable<Scalar<Type>>
    {
        public Scalar(string functionName, params IExpr[] inputs) : base(functionName, inputs) { }
        public Scalar(string functionName, IReadOnlyList<IExpr> inputs): base(functionName, inputs) { }

        public Backpropagation Backpropagation { get; internal set; }

        public virtual bool Equals(Type y) => false;

        public abstract void Backward(Scalar<Type> delta, Backpropagation bp);

        public override string ToString() => Name ?? InlineCodeGenerator.GetCode(this);

        public static implicit operator Scalar<Type>(Type v) => Const.Create(v);

        public static explicit operator Type(Scalar<Type> v) => ((Const)v).Value;

        public static Scalar<Type> operator -(Scalar<Type> x) => Neg<Type>.Create(x);

        public static Scalar<Type> operator +(Scalar<Type> x, Scalar<Type> y) => Add<Type>.Create(x, y);

        public static Scalar<Type> operator -(Scalar<Type> x, Scalar<Type> y) => Sub<Type>.Create(x, y);

        public static Scalar<Type> operator *(Scalar<Type> x, Scalar<Type> y) => Mul<Type>.Create(x, y);

        public static Scalar<Type> operator /(Scalar<Type> x, Scalar<Type> y) => Div<Type>.Create(x, y);

        public static Binary operator >(Scalar<Type> x, Scalar<Type> y) => new Gt<Type>(x, y);

        public static Binary operator <(Scalar<Type> x, Scalar<Type> y) => y > x;

        public static Binary operator >=(Scalar<Type> x, Scalar<Type> y) => new GtEq<Type>(x, y);

        public static Binary operator <=(Scalar<Type> x, Scalar<Type> y) => y >= x;

        public Scalar<T> As<T>() => Cast<T, Type>.Create(this);

        public bool IsZero => this.IsConst(Numeric<Type>.Zero);
        public bool IsOne => this.IsConst(Numeric<Type>.One);
        public bool IsMinusOne => this.IsConst(Numeric<Type>.MinusOne);
        public bool IsTwo => this.IsConst(Numeric<Type>.Two);

        /// <remarks>This implementation only works while there is no inputs</remarks>
        public override IExpr Patch(Patch substitutions) => PatternMatching.GetOrElse(substitutions.TryGetValue, this, this);

        public override System.Type GetArgumentType() => typeof(Type);

        public class Const : Scalar<Type>, IConst
        {
            public readonly Type Value;

            public static Dictionary<Type, Const> Cache = new Dictionary<Type, Const>();
            public static Const Create(Type v)
            {
                Const result;
                if (Cache.TryGetValue(v, out result)) return result;
                result = new Const(v);
                Cache.Add(v, result);
                return result;
            }

            private Const(Type v): base("Const")
            {
                this.Value = v;
            }

            public static implicit operator Const(Type v) => Const.Create(v);
            public static implicit operator Type(Const v) => v.Value;

            public string Literal => Numeric.GetLiteral<Type>(this.Value);

            public override bool Equals(Type f) => this.Value.Equals(f);

            public override void Process(IProcessor processor) => processor.ProcessLiteral(this, this.Value);

            public override void Backward(Scalar<Type> delta, Backpropagation bp)
            {
            }
        }

        public abstract class Symbol : Scalar<Type>, ISymbol
        {
            public Symbol(string name): base("Symbol")
            {
                this.Name = name;
            }

            public override void Backward(Scalar<Type> delta, Backpropagation bp) {}
        }

        public class Var : Symbol, IVar<Type>
        {
            public Var(string name) : base(name) {}

            public override void Process(IProcessor processor) => processor.ProcessVar(this);
        }

        public class Shared : Symbol, IShared<Type>
        {
            public Shared(Type value, string name) :
                base(name)
            {
                this.Value = value;
            }

            public static implicit operator Type(Shared shared) => shared.Value;

            public Type Value
            {
                get { return Numeric.GetScalar<Type>(this.Name); }
                set { Numeric.SetScalar<Type>(this.Name, value); }
            }

            public override void Process(IProcessor processor) => processor.ProcessShared(this);
        }

        /// <summary>
        /// Represents a function taking several inputs and having one output.
        /// </summary>
        public abstract class NAry : Scalar<Type>
        {
            IExpr[] _inputs;
            protected object[] _extraInputs = null;

            /// <param name="functionName">The name of the associated runtime function</param>
            /// <param name="inputs">the inputs of this node</param>
            /// <param name="extraInputs">inputs that are not expressions (like a bool or a string), should be avoided</param>
            public NAry(string functionName, IExpr[] inputs, object[] extraInputs): base(functionName, inputs)
            {
                _inputs = inputs;
                _extraInputs = extraInputs;
            }

            public NAry(string functionName, params IExpr[] inputs) : this(functionName, inputs, null) { }

            public NAry(string functionName, IEnumerable<IExpr> inputs) : this(functionName, inputs.ToArray(), null) { }

            /// <summary>
            /// Creates another object with the same constructor, but different inputs.
            /// Used during the patching.
            /// </summary>
            /// <param name="inputs">The new inputs, length and types must match `this.Inputs`.</param>
            /// <returns>A new expression with the same type.</returns>
            public abstract Scalar<Type> Clone(IReadOnlyList<IExpr> inputs);

            public override sealed IExpr Patch(Patch substitutions)
            {
                Scalar<Type> result;
                if (substitutions.TryGetValue(this, out result)) return result;
                var patchInputs = _inputs.Patch(substitutions);
                if (patchInputs == _inputs) return this;
                result = Clone(patchInputs);
                substitutions.Add(this, result);
                return result;
            }

            public override sealed void Process(IProcessor processor)
            {
                processor.ProcessFunctionCall(this, FunctionName, _extraInputs);
            }
        }

        public class Unary : NAry
        {
            /// <summary> A simple operator with one input. </summary>
            /// <param name="functionName">Typehe functionName used by the compiler</param>
            /// <param name="x">the only input</param>
            /// <param name="dx">the derivative of this expression wrt x</param>
            /// <param name="extraInputs"></param>
            public Unary(string functionName, Scalar<Type> x, Func<Scalar<Type>, Scalar<Type>, Scalar<Type>> dx, object[] extraInputs = null) : base(functionName, new[] { x }, extraInputs)
            {
                this.Dx = dx;
            }

            public Scalar<Type> x => (Scalar<Type>)this.Inputs.First();

            /// <summary>(x, f(x)) => Df(x) / Dx </summary>
            public readonly Func<Scalar<Type>, Scalar<Type>, Scalar<Type>> Dx;
            /// <summary> Caches the derivate. </summary>
            Scalar<Type> _dx;

            public override void Backward(Scalar<Type> delta, Backpropagation bp) => bp.PushGradientTo(x, delta * (_dx = _dx ?? Dx(x, this)));

            public override Scalar<Type> Clone(IReadOnlyList<IExpr> inputs) => Clone((Scalar<Type>)inputs[0]);

            public Scalar<Type> Clone(Scalar<Type> x) => new Unary(FunctionName, x, Dx, _extraInputs);
        }

        public class Binary : NAry
        {
            public readonly Func<Scalar<Type>, Scalar<Type>, Scalar<Type>, Scalar<Type>> Dx, Dy;
            private Scalar<Type> _dx, _dy;

            public Binary(
                string name, Scalar<Type> x, Scalar<Type> y,
                Func<Scalar<Type>, Scalar<Type>, Scalar<Type>, Scalar<Type>> dx,
                Func<Scalar<Type>, Scalar<Type>, Scalar<Type>, Scalar<Type>> dy,
                object[] extraInputs = null
            ) : base(name, new[] { x, y }, extraInputs)
            {
                this.Dx = dx ?? notImplemented;
                this.Dy = dy ?? notImplemented;
            }

            public Scalar<Type> x => (Scalar<Type>)this.Inputs[0];
            public Scalar<Type> y => (Scalar<Type>)this.Inputs[1];

            public override sealed void Backward(Scalar<Type> delta, Backpropagation bp)
            {
                _dx = _dx ?? Dx(x, y, this);
                _dy = _dy ?? Dy(x, y, this);
                bp.PushGradientTo(x, delta * _dx);
                bp.PushGradientTo(y, delta * _dy);
            }

            public override sealed Scalar<Type> Clone(IReadOnlyList<IExpr> inputs) => Clone((Scalar<Type>)inputs[0], (Scalar<Type>)inputs[1]);
            public Binary Clone(Scalar<Type> x, Scalar<Type> y) => new Binary(this.FunctionName, x, y, this.Dx, this.Dy, this._extraInputs);

            private static Scalar<Type> notImplemented(Scalar<Type> x, Scalar<Type> y, Scalar<Type> f)
            {
                throw new NotImplementedException();
            }
        }
    }

    public static class ScalarExtenstions
    {
        public static bool IsConst<T>(this Scalar<T> thiz, T value) => thiz is Scalar<T>.Const c && c.Value.Equals(value);

        public static bool IsNeg<T>(this Scalar<T> thiz) => thiz.Check((Scalar<T>.Unary u) => u.FunctionName == "Neg");
    }
}
