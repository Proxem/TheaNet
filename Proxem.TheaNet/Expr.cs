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

using Dim = Proxem.TheaNet.Scalar<int>;
using Int = Proxem.TheaNet.Scalar<int>.Const;

namespace Proxem.TheaNet
{
    /// <summary>
    /// Mathematical expression, represented as a graph.
    /// </summary>
    public interface IExpr
    {
        /// <summary> Friendly name of this expression. </summary>
        string Name { get; }

        /// <summary> Additional information on this expression. </summary>
        string Comment { get; set; }

        /// <summary>A unique Id </summary>
        long Id { get; }

        /// <summary> The list of expressions needed for computing this one. </summary>
        IReadOnlyList<IExpr> Inputs { get; }

        string FunctionName { get; }

        /// <summary>
        /// Another implementation of the "visitor" pattern.
        /// Dispatches to one of:
        /// ProcessLiteral, ProcessVar, ProcessShared, ProcessArray, ProcessFunctionCall, ProcessFor, ProcessSlice, ProcessElementwise.
        /// </summary>
        void Process(IProcessor processor);

        /// <summary> Replaces an expression in a graph by another one.</summary>
        /// <param name="substitutions">Dictionary of expressions to replace. Also used as a cache.</param>
        /// <returns>The new expression.</returns>
        IExpr Patch(Patch substitutions);

        Type GetArgumentType();
    }

    public abstract class Expr : IExpr
    {
        public Expr(string functionName, params IExpr[] inputs) : this(functionName, (IReadOnlyList<IExpr>)inputs) { }

        public Expr(string functionName, IReadOnlyList<IExpr> inputs)
        {
            Id = System.Threading.Interlocked.Increment(ref IdGenerator);
            FunctionName = functionName;
            _inputs = inputs;
        }

        public string Comment { get; set; }

        private static long IdGenerator;
        public long Id { get; }

        private IReadOnlyList<IExpr> _inputs;
        virtual public IReadOnlyList<IExpr> Inputs => _inputs;

        public readonly string FunctionName;
        string IExpr.FunctionName => this.FunctionName;

        private string _name;
        public string Name
        {
            get
            {
                return _name;
            }
            set
            {
                CheckName(value);
                if (Name != null)
                    Trace.TraceInformation($"{Name} renamed to {value}.");
                _name = value;
            }
        }

        private void CheckName(string value)
        {
            // TODO: check name collision
            //if (value.StartsWith("_")) throw new Exception("Names should not start with underscore");
        }

        public abstract IExpr Patch(Patch substitutions);

        public abstract void Process(IProcessor processor);

        public abstract Type GetArgumentType();
    }


    public interface IExpr<T> : IExpr
    {
    }

    public interface IScalar : IExpr
    {
    }

    public interface ITensor : IExpr
    {
        // Shape.Length
        int NDim { get; }
        Dim[] Shape { get; }

        [IndexerName("Slice")]
        ITensor this[params XSlice[] slices] { get; }

        [IndexerName("Slice")]
        ITensor this[IReadOnlyList<XSlice> slices] { get; }

        [IndexerName("Slice")]
        ITensor this[params Tensor<int>[] indices] { get; }
    }

    public interface ITensorVar : IExpr, ITensor, IVar { }

    public class NamedObject
    {
        public string Name;
        public object Object;
    }

    public static class NamingExtension
    {
        public static NamedObject Named(this object o, string name) => new NamedObject { Name = name, Object = o };

        public static Scalar<T> Named<T>(this Scalar<T> s, string name)
        {
            s.Name = name;
            return s;
        }

        public static Tensor<T> Named<T>(this Tensor<T> t, string name)
        {
            t.Name = name;
            return t;
        }
    }

    public class Lambda
    {
        public IVar[] Vars;
        public IExpr Expr;
    }

    public static class ShapeExtension
    {
        /// <summary>
        /// Returns true if the both shapes have a chance to match at runtime
        /// </summary>
        private static bool CanBroadcastTo(this Dim[] shapeX, Dim[] shapeY)
        {
            if (shapeX.Length != shapeY.Length) return false;
            for (int i = 0; i < shapeX.Length; ++i)
            {
                if (!CanBroadcastTo(shapeX[i], shapeY[i]))
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Returns true if the both shapes have a chance to match at runtime
        /// </summary>
        public static bool CanBroadcastTo(this Dim dimX, Dim dimY)
        {
            if (dimX is Int shapeX && dimY is Int shapeY)
            {
                int x = shapeX.Value, y = shapeY.Value;
                if (x != y && x != 1 && y != 1) return false;
            }
            return true;
        }

        /// <summary>
        /// Returns true if the both shapes have a chance to match at runtime
        /// </summary>
        public static bool NeedBroadcast(this Dim dimX, Dim dimY)
        {
            if (dimX is Int shapeX)
            {
                if (dimY is Int shapeY)
                {
                    int x = shapeX.Value, y = shapeY.Value;
                    if (x != y && x != 1 && y != 1)
                        throw new RankException($"Can't broadcast {x} to {y}");
                    if (x == 1 && y != 1) return true;
                    return false;
                }

                if (shapeX.Value == 1)
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Returns true if the two shapes have a chance to match at runtime
        /// </summary>
        public static bool CanEqualTo(this Dim[] shapeX, Dim[] shapeY)
        {
            if (shapeX.Length != shapeY.Length) return false;
            for (int i = 0; i < shapeX.Length; ++i)
            {
                if (!CanEqualTo(shapeX[i], shapeY[i]))
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Returns true if the two dimensions will equal at runtime
        /// </summary>
        public static bool WillEqualTo(this Dim dimX, Dim dimY)
        {
            if (dimX is Int shapeX && dimY is Int shapeY)
            {
                int x = shapeX.Value, y = shapeY.Value;
                if (x == y) return true;
            }
            return ( Equivalences.ContainsKey(dimX) && (Equivalences[dimX].Contains(dimY) || Equivalences[dimX].Any(dimY.WillEqualTo)) )
                || ( Equivalences.ContainsKey(dimY) && (Equivalences[dimY].Contains(dimX) || Equivalences[dimY].Any(dimX.WillEqualTo)) )
                || ( dimX == dimY )
            ;
        }

        /// <summary>
        /// Returns true if the two shapes will equal at runtime
        /// </summary>
        public static bool WillEqualTo(this Dim[] shapeX, Dim[] shapeY)
        {
            if (shapeX.Length != shapeY.Length) return false;
            for (int a = 0; a < shapeX.Length; ++a)
                if (!shapeX[a].WillEqualTo(shapeY[a]))
                    return false;
            return true;
        }

        /// <summary>
        /// Returns true if the two shapes have a chance to equal at runtime
        /// </summary>
        public static bool CanEqualTo(this Dim dimX, Dim dimY)
        {
            if (dimX is Int shapeX && dimY is Int shapeY)
            {
                int x = shapeX.Value, y = shapeY.Value;
                if (x != y) return false;
            }
            return true;
        }

        public static Dictionary<Dim, HashSet<Dim>> Equivalences = new Dictionary<Dim, HashSet<Dim>>();

        private static bool _bound(Dim dimX, Dim dimY) =>
            Equivalences.ContainsKey(dimX)
            && Equivalences.ContainsKey(dimY)
            && Equivalences[dimX] == Equivalences[dimY]
        ;

        private static void _bind(Dim dimX, Dim dimY)
        {
            if (_bound(dimX, dimY)) return;

            Trace.WriteLine($"Detected constraint: {dimX} = {dimY}.");
            if(Equivalences.ContainsKey(dimX) && Equivalences.ContainsKey(dimY))
            {
                var eqX = Equivalences[dimX];
                var eqY = Equivalences[dimY];
                Trace.WriteLine($"Merging respective equivalences set of size {eqX.Count} and {eqY.Count}");
                eqX.UnionWith(eqY);
                foreach (var y in eqY)
                    Equivalences[y] = eqX;
            }
            else if (Equivalences.ContainsKey(dimX) && !Equivalences.ContainsKey(dimY))
                Equivalences[dimY] = Equivalences[dimX];
            else if (!Equivalences.ContainsKey(dimX) && Equivalences.ContainsKey(dimY))
                Equivalences[dimX] = Equivalences[dimY];
            else
            {
                var eqXY = new HashSet<Dim> { dimX, dimY };
                Equivalences[dimX] = eqXY;
                Equivalences[dimY] = eqXY;
            }
        }

        /// <summary>
        /// Register both expression as things that must be equals.
        /// The compiler is then allowed to choose which expression is used to compute the value.
        /// If one argument is a Const the other will be replaced by the Const, simplifying the expression as soon as possible.
        /// </summary>
        public static bool Bind(ref Dim dimX, ref Dim dimY)
        {
            if (dimX == dimY) return false;
            var shapeX = dimX as Int;
            var shapeY = dimY as Int;
            if (shapeX != null && shapeY != null)
            {
                int x = shapeX.Value, y = shapeY.Value;
                if (x != y)
                    throw new RankException($"Can't bind {x} to {y}");
                return false;
            }
            _bind(dimX, dimY);
            // if dimX or dimY is a const, we use the const, to avoid later treatment
            if (shapeX != null)
                dimY = shapeX;
            else if (shapeY != null)
                dimX = shapeY;
            return true;
        }

        public static Dim[] DropAt(this IReadOnlyList<Dim> shape, int i, int n = 1)
        {
            if (i < 0) i += shape.Count;
            if (shape.Count >= i + n)
            {
                var res = new Dim[shape.Count - n];
                for (int a = 0; a < i; ++a)
                    res[a] = shape[a];
                for (int a = i + n; a < shape.Count; ++a)
                    res[a - n] = shape[a];
                return res;
            }
            else
                throw new RankException($"Can't drop {n} axes at index {i} from a {shape.Count} axes tensor");
        }

        public static Dim[] DropLeft(this IReadOnlyList<Dim> shape, int i)
        {
            if (shape.Count >= i)
                return shape.Skip(i).ToArray();
            //else if (axes.Count == i)
            //    return new Axis[] { new Axis { i = 1 } };
            else
                throw new RankException($"Can't drop {i} axes from a {shape.Count} axes tensor");
        }

        public static Dim[] DropRight(this IReadOnlyList<Dim> shape, int i)
        {
            if (shape.Count >= i)
                return shape.Take(shape.Count - i).ToArray();
            //else if (axes.Count == i)
            //    return new Axis[] { 1 };
            else
                throw new RankException($"Cant drop {i} axes from a {shape.Count} axes tensor");
        }

        public static Dim[] Pad(this Dim[] shape, Dim dim)
        {
            var padded = new Dim[shape.Length + 1];
            Array.Copy(shape, padded, shape.Length);
            padded[shape.Length] = dim;
            return padded;
        }

        public static Dim[] Pad(this Dim dim, Dim[] shape)
        {
            var padded = new Dim[shape.Length + 1];
            padded[0] = dim;
            Array.Copy(shape, 0, padded, 1, shape.Length);
            return padded;
        }

        public static Dim[] ToShape(this int[] shape)
        {
            return shape.Select(i => (Dim) i).ToArray();
        }

        public static string Format<T>(this Dim[] shape, IExpr<T> target)
        {
            var generator = new InlineCodeGenerator(null);
            generator.ObsoleteProcessArray<T, int>(target, shape);
            return generator.Result;
        }

        public static R[] Apply<T, R>(this T[] values, Func<T, R> f, R[] result = null)
        {
            result = result ?? new R[values.Length];
            for (int _ = 0; _ < values.Length; ++_)
                result[_] = f(values[_]);
            return result;
        }

        // polymorphism in C# at its finest
        public static T[] Patch<T>(this T[] exprs, Patch substitutions)
            where T : class, IExpr
        {
            bool changed;
            var patched = _patch(exprs, substitutions, out changed);
            return !changed ? exprs : patched;
        }

        public static IReadOnlyList<T> Patch<T>(this IReadOnlyList<T> exprs, Patch substitutions)
            where T : class, IExpr
        {
            bool changed;
            var patched = _patch(exprs, substitutions, out changed);
            return !changed ? exprs : patched;
        }

        public static IList<T> Patch<T>(this IList<T> exprs, Patch substitutions)
            where T : class, IExpr
        {
            bool changed;
            var patched = _patch(exprs, substitutions, out changed);
            return !changed ? exprs : patched;
        }

        private static T[] _patch<T>(IEnumerable<T> exprs, Patch substitutions, out bool changed)
            where T : class, IExpr
        {
            var patched = exprs.Select(i =>
                i?.Patch(substitutions)
            ).Cast<T>().ToArray();
            changed = (exprs, patched).Zip().Any(xy => xy.Item1 != xy.Item2);
            return patched;
        }

        public static T Where<T, U>(this T expr, Scalar<U> input, Scalar<U> equals) where T : IExpr =>
            (T)expr.Patch(new Patch { [input] = equals });

        public static T Where<T, U>(this T expr, Tensor<U> input, Tensor<U> equals) where T : IExpr =>
            (T)expr.Patch(new Patch { [input] = equals });

    }

    public interface IIndexable<Type>
    {
        Type this[params Scalar<int>[] indices] { get; }
    }

    public interface IScalar<T> : IExpr<T>, IScalar
    {
    }

    public interface ITensor<T> : IExpr<Array<T>>, ITensor
    {
    }

    public interface IDifferentiable<T>
    {
        void Backward(T delta, Backpropagation bp);
    }

    public interface ISymbol : IExpr {}

    public interface IConst : IExpr
    {
        string Literal { get; }
    }

    public interface IVar : ISymbol
    {
    }

    public interface IVar<T> : IVar, IExpr<T>
    {
    }

    public interface IShared : ISymbol
    {
    }

    public interface IShared<T> : IShared, IExpr<T>
    {
        T Value { get; set; }
    }

    /// <summary>
    /// A sequence computed recursively inside a loop
    /// </summary>
    public interface IFor : IExpr, ITensor
    {
        /// <summary> The loop containing this For </summary>
        Loop Loop { get; }

        /// <summary> The expression allowing to compute the next value of the sequence  </summary>
        ITensor Expression { get; set; }

        /// <summary> The seed value of the sequence, may be null </summary>
        ITensor OutputInfo { get; }

        ITensorVar RecursiveVariable { get; }

        bool IsRecursive { get; }

        // The index of the expression the loop => TODO: remove
        int Index { get; }
    }

    /// <summary>
    /// A set of sequences computed together from other sequences
    /// </summary>
    public interface ILoop
    {
        Scalar<int> Length { get; }

        /// <summary> The recursive variables used inside the loop </summary>
        IEnumerable<ITensorVar> Variables { get; }

        /// <summary> The recursive variables used inside the loop to describe a sequence or a recursive 'For' </summary>
        ITensorVar Variable(ITensor seqOrFor);

        /// <summary> The sequences used as input of the loop </summary>
        IEnumerable<ITensor> Sequences { get; }

        /// <summary> The Fors containing the expression to be computed </summary>
        IEnumerable<IFor> Fors { get; }
    }

    public interface IProcessor
    {
        void ProcessLiteral<T>(IExpr<T> target, T value);
        void ProcessVar<T>(IVar<T> target);
        void ProcessShared<T>(IShared<T> target);
        void ProcessList<T, U>(XList<T, U> target) where T : class, IExpr<U>;
        void ProcessFunctionCall<T>(IExpr<T> target, string name, object[] extras = null);
        void ProcessFor<T>(Tensor<T>.For target);
        void ProcessSlice(XSlice target);
        void ProcessElementwise<T>(Tensor<T>.Elementwise target);
    }
}