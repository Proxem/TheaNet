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
using System.Linq;
using System.Runtime.CompilerServices;

using Proxem.NumNet;
using Proxem.TheaNet.Binding;
using Proxem.TheaNet.Operators.IntScalars;
using Proxem.TheaNet.Operators.Tensors;

using Dim = Proxem.TheaNet.Scalar<int>;
using Scalars = Proxem.TheaNet.Operators.Scalars;

namespace Proxem.TheaNet
{
    [DebuggerDisplay("{_debugString}")]
    public abstract partial class Tensor<Type> : Expr, IExpr<Array<Type>>, ITensor<Type>, IDifferentiable<Tensor<Type>>, IIndexable<Scalar<Type>>
    {
        public Tensor(string functionName, params IExpr[] inputs) : base(functionName, inputs) { }
        public Tensor(string functionName, IReadOnlyList<IExpr> inputs): base(functionName, inputs) { }

        public readonly HashSet<string> Tags = new HashSet<string>();

        public void Tag(string tag) => Tags.Add(tag);

        public int NDim => Shape.Length;

        //[DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public abstract Dim[] Shape { get; }

        private Scalar<int> _size = null;
        public Scalar<int> Size => _size = (_size ?? Op.Size(Shape));

        protected Dim _runtimeShapeOf(int axis) => new AxisLength(this, axis);

        public abstract void Backward(Tensor<Type> delta, Backpropagation bp);

        public override string ToString() => InlineCodeGenerator.GetCode(this);

        private string _debugString
        {
            get
            {
                int max_length = 50;
                var shape = $"<{string.Join(", ", Shape.Select(s => s.ToString()))}>";
                var content = this.ToString();
                if (content.Length > max_length)
                    content = content.Substring(0, max_length) + " (...) ";
                return content + shape;
            }
        }

        [IndexerName("Slice")]
        public Tensor<Type> this[params XSlice[] slices] => Slicing<Type>.Create(this, slices);

        [IndexerName("Slice")]
        public Tensor<Type> this[params Slice[] slices] => Slicing<Type>.Create(this, slices.Select(s => (XSlice)s).ToList());

        [IndexerName("Slice")]
        public Tensor<Type> this[IReadOnlyList<XSlice> slices] => Slicing<Type>.Create(this, slices);

        [IndexerName("Slice")]
        public Tensor<Type> this[params Tensor<int>[] indices] => Indexing<Type>.Create(this, indices);

        ITensor ITensor.this[params XSlice[] slices] => this[slices];
        ITensor ITensor.this[IReadOnlyList<XSlice> slices] => this[slices];
        ITensor ITensor.this[params Tensor<int>[] indices] => this[indices];

        //[DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public IIndexable<Scalar<Type>> Item => this;

        Scalar<Type> IIndexable<Scalar<Type>>.this[params Scalar<int>[] indices] => Operators.Scalars.Item<Type>.Create(this, indices);

        public Tensor<Type> Reshape(params Scalar<int>[] shape) => Reshaping<Type>.Create(this, shape);

        public Tensor<Type> DimShuffle(params int[] axesPerm)
        {
            // axes may contain 'x' (120). Hopefuly no tensor will ever have so many dimensions...
            var trans = Transpose<Type>.Create(this, axesPerm.Where(a => a != 'x').ToArray());

            if (axesPerm.Contains('x'))
            {
                var shape = new Dim[axesPerm.Length];
                for (int d = 0; d < axesPerm.Length; ++d)
                {
                    if (axesPerm[d] == 'x') shape[d] = 1;
                    else if (axesPerm[d] < 0) shape[d] = Shape[NDim + axesPerm[d]];
                    else shape[d] = Shape[axesPerm[d]];
                }
                trans = trans.Reshape(shape);
            }

            return trans;
        }

        public static Tensor<Type> operator -(Tensor<Type> x)
        {
            switch (x)
            {
                case Elementwise unary when unary.Abstraction is Scalars.Neg<Type>:
                    return unary.Inputs[0];        // -(-x) = x
                case Elementwise binary when binary.Abstraction is Scalars.Sub<Type>:
                    return binary.Inputs[1] - binary.Inputs[0];    // -(x - y) = y - x
                case Fill<Type> fill:
                    return Op.Const(-fill.x, fill.Shape);
                case OneHot<Type> oneHot:
                    return Op.OneHot(oneHot.Shape, oneHot.Index, -oneHot.Content);
                default:
                    return Op.Apply(x, _x => -_x);
            }
        }

        public static Tensor<Type> operator +(Tensor<Type> x, Tensor<Type> y)
        {
            if (x is Elementwise unaryx && unaryx.Abstraction is Scalars.Neg<Type>)
                return y - unaryx.Inputs[0];    // (-x) + y = y - x

            if (y is Elementwise unaryy && unaryy.Abstraction is Scalars.Neg<Type>)
                return x - unaryy.Inputs[0];   // x + (-y) = x - y

            if (x.IsZero) return y;
            if (y.IsZero) return x;

                //() => x.Match((Neg<Type> neg) => y - neg.x),
                //() => y.Match((Neg<Type> neg) => x - neg.x),
            return Op.Apply(x, y, (_x, _y) => _x + _y);
        }

        public static explicit operator Scalar<Type>(Tensor<Type> t)
        {
            t.AssertOfDim(0);
            return Op.Sum(t);
        }

        public Tensor<Type> BroadcastTo(params Dim[] shape) => BroadCast<Type>.Create(this, shape);

        public static Tensor<Type> operator +(Tensor<Type> x, ITensor y)        // TODO: internal Add() ?
        {
            if (y is Tensor<Type> yt) return x + yt;
            throw new InvalidCastException($"Cannot convert type {x.GetType().GetName()} to type {y.GetType().GetName()}");
        }

        public static Tensor<Type> operator +(Scalar<Type> x, Tensor<Type> y) => Op.Apply(y, _y => x + _y);

        public static Tensor<Type> operator +(Tensor<Type> x, Scalar<Type> y) => y + x;

        static Dictionary<string, Tensor<Type>> Cache = new Dictionary<string, Tensor<Type>>();
        public static Tensor<Type> operator -(Tensor<Type> x, Tensor<Type> y)
        {
            return Cache.GetValue($"Sub|{x.Id}|{y.Id}", () =>
            {
                if (x == y) return Op.ZerosLike(x);

                if (x is Elementwise unaryx && unaryx.Abstraction is Scalars.Neg<Type>)
                    return -(unaryx.Inputs[0] + y);        // (-x) - y = -(x + y)

                if (y is Elementwise unaryy && unaryy.Abstraction is Scalars.Neg<Type>)
                    return x + unaryy.Inputs[0];          // x - (-y) = x + y

                return Op.Apply(x, y, (_x, _y) => _x - _y);

                //return Option.TakeFirst(
                //    () => x.Match((Neg<Type> neg) => -(neg.x + y)),
                //    () => y.Match((Neg<Type> neg) => x + neg.x),
                //    () => new BinaryElementwise(x, y, (_x, _y) => new Operators.Scalars.Sub<Type>(_x, _y))
                //);
            });
        }

        public static Tensor<Type> operator -(Tensor<Type> x, Scalar<Type> y) => x + (-y);

        public static Tensor<Type> operator -(Scalar<Type> x, Tensor<Type> y) => Op.Apply(y, _y => x - _y);

        public static Tensor<Type> operator *(Tensor<Type> x, Tensor<Type> y)
        {
            if (x.IsZero) return x;
            else if (x.IsOne) return y;
            else if (x.IsMinusOne) return -y;
            else if (y.IsZero) return y;
            else if (y.IsOne) return x;
            else if (y.IsMinusOne) return -x;

            if (x is Fill<Type> fillx && y is Fill<Type> filly)
                return Op.Const(fillx.x * filly.x, fillx.Shape);

            if (x == y)
                return Op.Square(x);

            // this optimizations are bad because they assume that all abstractions are of the form (x op y) which is not the case
            // TODO: replace them
            if (x is Elementwise elementwisex)
            {
                if (elementwisex.Abstraction is Operators.Scalars.Div<Type> div)
                {
                    if (elementwisex.Inputs.Length == 1 && elementwisex.Inputs[0] == y)
                    {
                        if (div.Inputs[0] is Scalar<Type>.Const c)
                        {
                            return Fill<Type>.Create(c, y.Shape);
                        }
                    }
                    //if (elementwisex.Inputs[1] == y) return elementwisex.Inputs[0];   // Mul(Div(x, y), y) = x
                }

                if (elementwisex.Abstraction is Scalar<Type>.Binary binary && binary.FunctionName == "Div")
                {
                    if (elementwisex.Inputs[0] == y) return Fill<Type>.Create(binary.x, y.Shape);   // Mul(Div(x, y), y) = x
                }
            }

            //var binaryy = y as Elementwise;
            //if (binaryy != null)
            //{
            //    var mul = binaryy.Abstraction as Operators.Scalars.Mul<Type>;
            //    if (mul != null) return (x * binaryy.Inputs[0]) * binaryy.Inputs[1];                // hinders common subexpressions detection ?

            //    var div = binaryy.Abstraction as Operators.Scalars.Div<Type>;
            //    if (div != null)
            //    {
            //        if (binaryy.Inputs[0] is Fill<Type>) return binaryy.Inputs[0] * (x / binaryy.Inputs[1]);
            //        if (binaryy.Inputs[1] == x) return binaryy.Inputs[0];   // Mul(x, Div(y, x)) = y
            //    }
            //}

            return PatternMatching.TakeFirst<Tensor<Type>>(
                //() => y.Match((Mul<Type> mul) => (x * mul.x) * mul.y),                // hinders common subexpressions detection ?
                () => x.Match((OneHot<Type> oneHot) => Op.OneHot(oneHot.Shape, oneHot.Index, oneHot.Content * y[oneHot.Index])),
                () => y.Match((OneHot<Type> oneHot) => Op.OneHot(oneHot.Shape, oneHot.Index, oneHot.Content * x[oneHot.Index])),
                () => x.Match((OneHotSlice<Type> oneHot) => Op.OneHot(oneHot.Shape, oneHot.Slices, oneHot.Content * y[oneHot.Slices])),
                () => y.Match((OneHotSlice<Type> oneHot) => Op.OneHot(oneHot.Shape, oneHot.Slices, oneHot.Content * x[oneHot.Slices])),
                () => x.Match((OneHotPoint<Type> oneHot) => Op.OneHot(oneHot.Shape, oneHot.Indexes, oneHot.Content * y.Item[oneHot.Indexes])),
                () => y.Match((OneHotPoint<Type> oneHot) => Op.OneHot(oneHot.Shape, oneHot.Indexes, oneHot.Content * x.Item[oneHot.Indexes])),
                //() => y.Match((Inv<Type> inv) => inv.a * (x / inv.x)),
                //() => x.Match((Div<Type> div) => div.y == y ? div.x : new Mul<Type>(x, y)),     // Mul(Div(x, y), y) = x
                //() => y.Match((Div<Type> div) => div.y == x ? div.x : new Mul<Type>(x, y)),     // Mul(x, Div(y, x)) = y
                () => Op.Apply(x, y, (_x, _y) => _x * _y)
            );
        }

        public static Tensor<Type> operator *(Scalar<Type> x, Tensor<Type> y) => y.Match(
            (OneHot<Type> onehot) => Op.OneHot(onehot.Shape, onehot.Index, x * onehot.Content),
            (OneHotPoint<Type> onehot) => Op.OneHot(onehot.Shape, onehot.Indexes, x * onehot.Content),
            (OneHotSlice<Type> onehot) => Op.OneHot(onehot.Shape, onehot.Slices, x * onehot.Content),
            () => Op.Apply(y, _y => x * _y)
        );

        public static Tensor<Type> operator *(Tensor<Type> x, Scalar<Type> y) => y * x;

        public static Tensor<Type> operator /(Tensor<Type> x, Tensor<Type> y)
        {
            x.AssertOfDim(y.NDim);
            if (x == y) return Op.OnesLike(x);
            if (x.IsZero) return Op.ZerosLike(y);
            if (y.IsOne) return x;
            if (x is Fill<Type> fx && y is Fill<Type> fy)
                return Op.Const(fx.x / fy.x, y.Shape);

            if (y is Elementwise unaryy && unaryy.Abstraction is Scalars.Neg<Type>)
                return -x / unaryy.Inputs[0];      // x / (-y) = -(x / y)

            //() => y.Match((Neg<Type> neg) => -(x / neg.x)),
            return Op.Apply(x, y, (_x, _y) => new Operators.Scalars.Div<Type>(_x, _y));
        }

        public static Tensor<Type> operator /(Tensor<Type> x, Type y)
        {
            if (x is Tensor<int>)
                return Op.Apply(x, _x => _x / y);
            else
                return x * Numeric.Div(Numeric<Type>.One, y);
        }

        public static Tensor<Type> operator /(Tensor<Type> x, Scalar<Type> y) => Op.Apply(x, _x => _x / y);

        public static Tensor<Type> operator /(Scalar<Type> x, Tensor<Type> y) => Op.Apply(y, _y => x / _y);

        public static Tensor<Type> operator >(Tensor<Type> a, Scalar<Type>.Const b) => Op.Apply(a, a_ => new Operators.Scalars.GtCst<Type>(a_, b));

        public static Tensor<Type> operator >(Tensor<Type> a, Type b) => a > Op.Const(b);

        public static Tensor<Type> operator <(Tensor<Type> a, Type b) => a < Op.Const(b);

        public static Tensor<Type> operator <(Tensor<Type> a, Scalar<Type>.Const b)
        {
            throw new NotImplementedException();
        }

        public static Tensor<Type> operator >(Tensor<Type> a, Tensor<Type> b) => Op.Apply(a, b, (a_, b_) => a_ > b_);

        public static Tensor<Type> operator >=(Tensor<Type> a, Tensor<Type> b) => Op.Apply(a, b, (a_, b_) => a_ >= b_);

        public static Tensor<Type> operator <(Tensor<Type> a, Tensor<Type> b) => b > a;

        public static Tensor<Type> operator <=(Tensor<Type> a, Tensor<Type> b) => b >= a;

        public Tensor<T> As<T>() => new Cast<T, Type>(this);

        public bool IsConst(Type value) => this is Fill<Type> f && f.x.IsConst(value);

        //[DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public bool IsZero => IsConst(Numeric<Type>.Zero);

        //[DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public bool IsOne => IsConst(Numeric<Type>.One);

        //[DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public bool IsMinusOne => IsConst(Numeric<Type>.MinusOne);

        //[DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public bool IsTwo => IsConst(Numeric<Type>.Two);

        public abstract class Symbol : Tensor<Type>, ISymbol
        {
            public Symbol(Dim[] shape, string name): base("TSymbol")
            {
                Name = name;
                Shape = shape;
            }

            public override Dim[] Shape { get; }

            public override IExpr Patch(Patch substitutions)
            {
                Tensor<Type> result;
                if (substitutions.TryGetValue(this, out result))
                    return result;
                else if(this is Var)
                {
                    var variable = this as Var;
                    var patchShape = new Dim[NDim];
                    bool changed = false;
                    for(int a=0; a < NDim; ++a)
                    {
                        if (variable.AxisIsBounded(a))
                        {
                            patchShape[a] = (Dim)Shape[a].Patch(substitutions);
                            changed = changed || patchShape[a] != Shape[a];
                        }
                        else
                            patchShape[a] = -1;
                    }
                    if (changed)
                        result = new Var(patchShape, Name);
                    else
                        result = this;
                }
                else
                    result = this;
                substitutions.Add(this, result);
                return result;
            }

            public override void Backward(Tensor<Type> delta, Backpropagation bp) { }
        }

        public class Var : Symbol, IVar<Array<Type>>, ITensorVar
        {
            public Var(Dim[] shape, string name) :
                base(shape, name)
            {
                for (int d = 0; d < NDim; ++d)
                    if (Shape[d] is Scalar<int>.Const && (Shape[d] as Scalar<int>.Const).Value < 0)
                        Shape[d] = _runtimeShapeOf(d);
            }

            public Var(int NDim, string name) :
                base(new Dim[NDim], name)
            {
                for (int d = 0; d < NDim; ++d)
                    Shape[d] = _runtimeShapeOf(d);
            }

            public bool AxisIsBounded(int axis) => !(Shape[axis] is AxisLength length && length.x == this);

            public override void Process(IProcessor processor) => processor.ProcessVar(this);
        }

        public class Shared : Symbol, IShared<Array<Type>>
        {
            // TODO: borrow = false => deep copy v (default in theano)
            public Shared(Array<Type> value, string name) :
                base(value.Shape.ToShape(), name)
            {
                this.Value = value;
            }

            public Array<Type> Value
            {
                get { return Numeric.GetTensor<Type>(this.Name); }
                set { Numeric.SetTensor(this.Name, value); }
            }

            public override void Process(IProcessor processor) => processor.ProcessShared(this);
        }

        /// <summary>
        /// Represents a function taking several inputs and having one array output.
        /// </summary>
        public abstract class NAry : Tensor<Type>
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

            public abstract NAry Clone(IReadOnlyList<IExpr> inputs);

            public override sealed IExpr Patch(Patch substitutions)
            {
                Tensor<Type> result;
                if (substitutions.TryGetValue(this, out result)) return result;
                var patchInputs = _inputs.Patch(substitutions);
                if (patchInputs == _inputs) return this;
                result = Clone(patchInputs);
                substitutions.Add(this, result);
                return result;
            }

            public override void Process(IProcessor processor) => processor.ProcessFunctionCall(this, FunctionName, _extraInputs);
        }

        public abstract class Unary<U, U_> : NAry
            where U : IExpr<U_>
        {
            public Unary(string functionName, U x): base(functionName, x) { }
            public Unary(string functionName, U x, params object[] extraInputs): base(functionName, new IExpr[] { x }, extraInputs) { }

            public U x => (U)Inputs[0];

            public sealed override NAry Clone(IReadOnlyList<IExpr> inputs) => Clone((U)inputs[0]);

            public abstract Unary<U, U_> Clone(U x);
        }

        public abstract class Binary<U, U_, V, V_> : NAry
                where U : IExpr<U_>
                where V : IExpr<V_>
        {
            public Binary(string functionName, U x, V y): base(functionName, x, y) { }

            public Binary(string functionName, U x, V y, params object[] extraInputs): base(functionName, new IExpr[] { x, y }, extraInputs) { }

            public U x => (U)Inputs[0];

            public V y => (V)Inputs[1];

            public sealed override NAry Clone(IReadOnlyList<IExpr> inputs) => Clone((U)inputs[0], (V)inputs[1]);

            public abstract Binary<U, U_, V, V_> Clone(U x, V y);
        }

        /// <summary>An aggregate is a Tensor that reduces the dimensionnality of its input.</summary>
        public abstract class Aggregate : Unary<Tensor<Type>, Array<Type>>
        {
            public readonly int Axis;

            /// <param name="functionName"></param>
            /// <param name="x">The input.</param>
            /// <param name="axis">The axis that will be reduced (`Aggregate(x, axis).Shape[axis] == 1`)</param>
            public Aggregate(string functionName, Tensor<Type> x, int axis) : base(functionName, x, axis.Named(nameof(axis)))
            {
                this.Axis = axis < 0 ? axis + x.NDim : axis;
                if (Axis < 0 || Axis >= x.NDim)
                    throw new IndexOutOfRangeException($"Can't aggregate on axis {axis}. {x} has only {x.NDim} axes.");
                Shape = new Dim[x.NDim];
                x.Shape.CopyTo(Shape, 0);
                Shape[Axis] = 1;
            }

            public override sealed Dim[] Shape { get; }

            public abstract Aggregate Clone(Tensor<Type> x, int axis);

            public override sealed Unary<Tensor<Type>, Array<Type>> Clone(Tensor<Type> x) => Clone(x, Axis);
        }
    }

    public static class DictionaryExtensions
    {
        public static V GetValue<K, V>(this Dictionary<K, V> dictionary, K key, Func<V> defaultValue)
        {
            V result;
            if (dictionary.TryGetValue(key, out result)) return result;
            result = defaultValue();
            dictionary.Add(key, result);
            return result;
        }
    }
}
