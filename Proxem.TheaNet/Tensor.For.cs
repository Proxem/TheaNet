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

using Proxem.NumNet;
using Proxem.TheaNet.Binding;
using Proxem.TheaNet.Operators.Tensors;

using static Proxem.TheaNet.LoopNamer;
using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet
{
    partial class Tensor<Type>
    {
        public class For : Tensor<Type>, IFor
        {
            public readonly Loop Loop;       // A loop containing several expressions
            public readonly int Index;           // The index of the expression the loop => TODO: remove
            public readonly Tensor<Type> OutputInfo;
            public readonly Var RecursiveVariable;

            private Tensor<Type> _expression;
            private Dim[] _shape;

            internal For(Loop loop, int index, Tensor<Type> expression, Tensor<Type> outputInfo, Var variable): base("For", null)
            {
                this.Loop = loop;
                this.Index = index;
                this.OutputInfo = outputInfo;
                this.RecursiveVariable = variable;

                // if expression is null, will do nothing
                this.Expression = expression;
                this.Name = loop.Name + "_" + (variable?.Name ?? loop.Fors.Count.ToString());
            }
            /// <summary>
            /// The expression used to compute a new element of this For. The setter also set the shape.
            /// </summary>
            public Tensor<Type> Expression
            {
                get { return _expression; }
                internal set
                {
                    if (value != null)
                    {
                        _expression = value;
                        _shape = Loop.Length.Pad(_expression.Shape);
                        if (OutputInfo != null)
                            OutputInfo.AssertOfShape(_expression);
                    }
                }
            }

            public override IReadOnlyList<IExpr> Inputs => Loop.GetInputs();

            public override Dim[] Shape => _shape;

            ITensor IFor.Expression
            {
                get { return this.Expression; }
                set { this.Expression = (Tensor<Type>)value; }
            }

            Loop IFor.Loop => this.Loop;

            ITensor IFor.OutputInfo => OutputInfo;

            ITensorVar IFor.RecursiveVariable => RecursiveVariable;

            int IFor.Index => Index;

            public bool IsRecursive => this.OutputInfo != null;

            public override IExpr Patch(Patch substitutions)
            {
                Tensor<Type> result;
                if (substitutions.TryGetValue(this, out result)) return result;
                var patchedLoop = Loop.Patch(substitutions);
                return patchedLoop.Fors[Index];
            }

            public override void Process(IProcessor processor) => processor.ProcessFor(this);

            public override void Backward(Tensor<Type> deltas, Backpropagation bp)
            {
                deltas.AssertOfShape(Shape);

                var deltaFromRecursive = OutputInfo != null;

                // var in the forward -> for in the backward
                var forsDic = new Dictionary<ISymbol, IFor>();   // ITensorSymbol

                var backLoop = new Loop("d" + Loop.Name);
                backLoop.Length = Loop.Length;
                var substitution = new Patch(preserveShape: true);

                // add the sequences used by the forward
                int fwdSeqCount = Loop.Sequences.Count;
                for (int i = 0; i < fwdSeqCount; i++)
                {
                    var seq = Loop.Sequences[i];
                    var variable = Loop.Variable(seq);
                    var alias = Loop.Sequences[i].Match(
                        (Tensor<float> s) =>
                            backLoop.AddSeq(s[Step_m1], variable.Name + "_", Loop.SequenceAxes[i]),
                        (Tensor<int> s) =>
                            backLoop.AddSeq(s[Step_m1], variable.Name + "_", Loop.SequenceAxes[i]),
                        (Func<ITensor>)null
                    );
                    substitution.Add_(variable, alias);
                }

                // add the sequences computed by the forward
                foreach (var @for in Loop.Fors)
                {
                    if (@for.IsRecursive)
                    {
                        var variable = @for.RecursiveVariable;
                        var alias = @for.Match(
                            (Tensor<float>.For f) =>
                                backLoop.AddSeq(new Insert<float>(f, 0, f.OutputInfo, 0)[From_m2_Step_m1], variable.Name + "_", axis: 0),
                            (Tensor<int>.For f) =>
                                backLoop.AddSeq(new Insert<int>(f, 0, f.OutputInfo, 0)[From_m2_Step_m1], variable.Name + "_", axis: 0),
                            (Func<ITensor>)null
                        );
                        substitution.Add_(variable, alias);
                    }
                    else
                    {
                        var alias = @for.Match(
                            (Tensor<float>.For f) =>
                                backLoop.AddSeq(f[Step_m1], @for.Name + "_"),
                            (Tensor<int>.For f) =>
                                backLoop.AddSeq(f[Step_m1], @for.Name + "_"),
                            (Func<ITensor>)null
                        );
                        substitution.Add_(@for.Expression, alias);
                    }
                }

                // add the retropropagated delta
                var deltaOut = backLoop.AddSeq(deltas[Step_m1], $"delta_{RecursiveVariable?.ToString() ?? "f" + Index}_", axis: 0);

                // d_ avoid duplicated variables with the same name.
                var d_ = new Dictionary<IVar, IVar>();

                // add deltas of sequences (inputs of and computed by the forward), initialized to zero
                var recVariables = Loop.RecursiveFors.Select(f => Loop.Variable(f));
                foreach (var varFwd in Loop.Variables)
                {
                    var zeros = varFwd.Match(
                        (Tensor<float>.Var x) => Op.ZerosLike(x),
                        (Tensor<int>.Var x) => Op.ZerosLike(x),
                        (Func<ITensor>)null
                    );
                    var @for = backLoop.AddRecursive_(zeros, zeros, $"d{varFwd.Name}_");
                    @for.Comment = $"dL/d{varFwd}";

                    d_[varFwd] = @for.RecursiveVariable;
                    forsDic[varFwd] = @for;
                }

                // `others` collect gradients pushed to expressions of the loop that aren't sequences or variables.
                var others = new Dictionary<IExpr, IFor>();
                AddDeltaFromBackpropagate(backLoop, others, forsDic, Backpropagation.Backward(Expression, deltaFromRecursive ? deltaOut + (Var)d_[RecursiveVariable] : deltaOut));

                foreach (var @for in Loop.RecursiveFors)
                {
                    var variable = @for.RecursiveVariable;

                    if (!deltaFromRecursive || @for != this)
                    {
                        var gradExpr = @for.Match(
                            (Tensor<float>.For f) => Backpropagation.Backward(f.Expression, (Tensor<float>)d_[f.RecursiveVariable]),
                            (Tensor<int>.For f) => Backpropagation.Backward(f.Expression, (Tensor<int>)d_[f.RecursiveVariable]),
                            null
                        );

                        AddDeltaFromBackpropagate(backLoop, others, forsDic, gradExpr);
                    }
                    // else: we already added the delta prior to the loop

                    // reuse results computed during the forward inside the backward
                    var alias_tp1 = backLoop.AddRecursive_(variable, @for[-1], variable.Name + "_tp1").RecursiveVariable;
                    substitution.Add_(@for.Expression, alias_tp1);
                }

                // Substitute variable in fors
                foreach (var @for in backLoop.Fors)
                {
                    var comment = @for.Expression.Comment;
                    @for.Expression = (ITensor)@for.Expression.Patch(substitution);
                    @for.Expression.Comment = comment;
                }

                // deltas of sequences
                for (int i = 0; i < Loop.Sequences.Count; ++i)
                    if (Loop.Sequences[i] is Tensor<float>)
                        bp.PushGradientTo((Tensor<float>)Loop.Sequences[i], ((Tensor<float>)backLoop.Fors[i])[Step_m1]);
                    else
                        throw new NotImplementedException();

                // deltas of seed
                foreach (var @for in Loop.RecursiveFors)
                    if(@for is Tensor<float>)
                        bp.PushGradientTo((Tensor<float>)@for.OutputInfo, ((Tensor<float>)forsDic[@for.RecursiveVariable])[-1]);
                    else
                        throw new NotImplementedException();

                // other deltas
                foreach (var W_dW in others)
                {
                    var W = W_dW.Key; var dW = W_dW.Value;
                    if (W is Tensor<float>)
                        bp.PushGradientTo((Tensor<float>)W, Op.Sum((Tensor<float>)dW, axis: 0));
                    else
                        throw new NotImplementedException();
                }
            }

            private void AddDeltaFromBackpropagate(Loop backLoop, Dictionary<IExpr, IFor> others, Dictionary<ISymbol, IFor> forsDic, Backpropagation bp)
            {
                foreach (var partial_i in bp.TensorDerivatives)
                {
                    if (!(partial_i.Key is ISymbol)) continue;
                    if (partial_i.Key is ITensorVar x && Loop.IsVariable(x)) // partial_i.Value = d(expr[j]) / d(partial_i.Key)
                    {
                        var dExpr_dx = partial_i.Value;
                        forsDic[x].Expression = _add(forsDic[x].Expression, dExpr_dx);
                        forsDic[x].Expression.Comment = $"dL/d{x}";
                    }
                    else
                    {
                        var W = partial_i.Key;
                        var dExpr_dW = partial_i.Value;
                        if (!others.ContainsKey(W))
                        {
                            dExpr_dW.Comment = $"dL/d{W}";
                            others[W] = backLoop.AddOutput_(partial_i.Value);
                        }
                        else
                        {
                            others[W].Expression = _add(others[W].Expression, partial_i.Value);
                        }
                    }
                }
            }
        }

        /// <summary> [::-1] </summary>
        private readonly static XList<XSlice, Slice> Step_m1 = new XList<XSlice, Slice>(new[] { XSlicer.Step(-1) });
        /// <summary> [-2::-1] </summary>
        private readonly static XList<XSlice, Slice> From_m2_Step_m1 = new XList<XSlice, Slice>(new[] { XSlicer.From(-2, step: -1) });

        private static ITensor _add(ITensor a, ITensor b)
        {
            if (a is Tensor<float>)
                return (Tensor<float>)a + (Tensor<float>)b;
            else if (a is Tensor<int>)
                return (Tensor<int>)a + (Tensor<int>)b;
            else
                throw new NotImplementedException();
        }
    }

    /// <summary>Helper class that generate loop names.</summary>
    public static class LoopNamer
    {
        private static int _f = 0;

        public static string LoopName() => "_loop_" + (_f++);
    }

    /// <summary>
    /// Represents a loop over a recursive function f(v0, ..., vn) => (e0,..., em)
    /// (v0, ..., vn) a stored in Variables
    /// (e0, ..., em) a stored in Expressions
    /// Invariants:
    ///     Variables.Count = Sequences.Count + OutputsInfo.Count(o => o != null)
    ///     Expressions.Count = OutputsInfo.Count
    /// </summary>
    public class Loop : ILoop
    {
        public Compiler Compiler;
        public readonly string Name;

        /// <summary>Map tensor (sequences and recursive) to the variable representing them inside the loop</summary>
        protected readonly Dictionary<ITensor, ITensorVar> _variables;
        /// <summary> The sequences used as input of the loop </summary>
        public readonly IList<ITensor> Sequences;
        /// <summary>For each sequence contains the axis along which it will be sliced</summary>
        public readonly IList<int> SequenceAxes;

        /// <summary> The Fors containing the expression to be computed </summary>
        public readonly IList<IFor> Fors;

        /// <summary>The length of the sequence we are iterating on</summary>
        public Dim Length { get; set; }

        public bool SequencesLocked { get; private set; }

        public IEnumerable<ITensorVar> Variables => this._variables.Values;

        public ITensorVar Variable(ITensor seqOrFor) => _variables[seqOrFor];
        public Tensor<T>.Var Variable<T>(Tensor<T> seqOrFor) => (Tensor<T>.Var)_variables[seqOrFor];
        ITensorVar ILoop.Variable(ITensor seqOrFor) => this.Variable(seqOrFor);

        IEnumerable<ITensor> ILoop.Sequences => this.Sequences;

        IEnumerable<IFor> ILoop.Fors => this.Fors;

        public IEnumerable<IFor> RecursiveFors => Fors.Where(@for => @for.RecursiveVariable != null);

        public IEnumerable<IFor> OutputedFors => Fors.Where(@for => @for.RecursiveVariable == null);

        private IReadOnlyList<IExpr> _inputs = null;

        /// <summary>
        /// The inputs of the loop in the following order: Length, Sequences, Expressions with their OutputInfo when not null
        /// Also locks the loop and prevent further modification.
        /// </summary>
        public IReadOnlyList<IExpr> GetInputs()
        {
            // TODO: I wanted Inputs to be a Property, but while debugging the loop risk to be locked by an evaluation of the property.
            _inputs = _inputs ?? _generateInputs().ToList();
            return _inputs;
        }

        /// <summary> Generates the list with all the inputs. As the list is ReadOnly, once it is created, the loop is locked. </summary>
        private IEnumerable<IExpr> _generateInputs()
        {
            SequencesLocked = true;
            // TODO: add RecursiveVariable
            yield return Length;

            foreach (var s in Sequences)
                yield return s;
            foreach (var f in Fors)
            {
                yield return f.Expression;
                if (f.OutputInfo != null)
                    yield return f.OutputInfo;
            }
        }

        public override string ToString() => $"{Name}: Scan({string.Join(", ", _variables)})";

        public Loop(string name, Delegate fn, IReadOnlyList<ITensor> sequences, IReadOnlyList<ITensor> outputsInfo, int axis) : this(name)
        {
            var parameters = fn.Method.GetParameters();
            var recCount = outputsInfo == null ? 0 : outputsInfo.Count(o => o != null);
            if (sequences.Count + recCount != parameters.Length)
                throw new ArgumentException($"Incoherent number of arguments: found {parameters.Length} in the lambda but was given {sequences.Count} sequences and {recCount} output infos.");

            // build variables from Sequences and corresponding fn's parameters
            int p = 0;
            foreach (dynamic sequence in sequences)
                AddSeq(sequence, parameters[p++].Name, axis);

            var recursiveVariables = new List<ITensorVar>();
            // = outputsInfo?.Zip(parameters.Skip(p), _createVar);

            if (outputsInfo != null)
            {
                // build variables from the recursive variables and corresponding fn's parameters
                foreach (var output in outputsInfo)
                {
                    if (output == null)
                        recursiveVariables.Add(null);
                    else
                        recursiveVariables.Add(_createVar(output, parameters[p++]));
                }
            }
            Debug.Assert(p == parameters.Length);

            // the sequence variables are already in Variables but the recursive aren't.
            var variables = Variables.Concat(recursiveVariables.Where(r => r != null)).ToArray();
            // build expressions by invoking lambda
            var fnResult = fn.DynamicInvoke(variables);
            IReadOnlyList<ITensor> expressions;
            if (fnResult is ITensor)
                expressions = new List<ITensor> { (ITensor)fnResult };
            else if (fnResult is IReadOnlyList<ITensor>)
                expressions = (IReadOnlyList<ITensor>)fnResult;
            else
                throw new ArgumentException($"Bad fn return type: expecting Tensor or IList<Tensor>, got {fnResult.GetType().Name}", nameof(fn));

            if (outputsInfo != null && expressions.Count != outputsInfo.Count)
                throw new ArgumentException($"Wrong outputsInfo count: expecting {expressions.Count} got {outputsInfo.Count}", nameof(outputsInfo));

            if (outputsInfo != null)
            {
                foreach (var (variable, expr, seed) in (recursiveVariables, expressions, outputsInfo).Zip())
                {
                    if (seed == null)
                        AddOutput_(expr);
                    else
                        AddRecursive_(expr, seed, variable);
                }
            }
            else
            {
                foreach (dynamic expr in expressions)
                    AddOutput(expr);
            }
        }

        private Loop(string name, Dictionary<ITensor, ITensorVar> variables, IList<ITensor> sequences, IList<int> sequenceAxes, IList<IFor> fors, Scalar<int> length)
        {
            this.Name = name;
            this._variables = variables;
            this.Sequences = sequences;
            this.SequenceAxes = sequenceAxes;
            this.Fors = fors;
            this.Length = length;
        }

        public Loop(string name, Scalar<int> length = null) :
            this(name, new Dictionary<ITensor, ITensorVar>(), new List<ITensor>(), new List<int>(), new List<IFor>(), length)
        { }

        /// <summary>
        /// Add a sequence to the loop.
        /// Returns the variable used internally to iterate over the sequence.
        /// This variable is named after the given `name`.
        /// </summary>
        public Tensor<T>.Var AddSeq<T>(Tensor<T> seq, string name, int axis = 0) =>
            AddSeq(seq, new Tensor<T>.Var(seq.Shape.DropAt(axis), name), axis);

        public Tensor<T>.Var AddSeq<T>(Tensor<T> seq, Tensor<T>.Var variable, int axis = 0)
        {
            if (SequencesLocked) throw new Exception("This loop is locked");

            Length = Length ?? seq.Shape[axis];

            Sequences.Add(seq);
            SequenceAxes.Add(axis);
            _variables.Add(seq, variable);
            return variable;
        }

        /// <summary>
        /// Add a recursive For to the loop.
        /// Returns the For created.
        /// The For variable is named after the given `name`.
        /// </summary>
        public Tensor<T>.For AddRecursive<T>(Tensor<T> expr, Tensor<T> outputInfo, string name) =>
            AddRecursive(expr, outputInfo, new Tensor<T>.Var(outputInfo.Shape, name));

        public Tensor<T>.For AddRecursive<T>(Tensor<T> expr, Tensor<T> outputInfo, Tensor<T>.Var recursive)
        {
            if (SequencesLocked) throw new Exception("This loop is locked");

            if (outputInfo == null)
                throw new ArgumentException("outputInfo can't be null, use AddOutput instead", nameof(outputInfo));

            var f = new Tensor<T>.For(this, Fors.Count, expr, outputInfo, recursive);
            Fors.Add(f);
            _variables.Add(f, recursive);
            return f;
        }

        /// <summary> Add a recursive For to the loop. Returns the For created. </summary>
        public IFor AddRecursive_(ITensor expr, ITensor seed, string name)
        {
            if (seed == null)
                throw new ArgumentException($"{nameof(seed)} can't be null, use AddOutput instead", nameof(seed));

            ITensorVar variable = seed.Match(
                (Tensor<float> exprF) => new Tensor<float>.Var(seed.Shape, name),
                (Tensor<int> exprI) => new Tensor<int>.Var(seed.Shape, name),
                () => { throw new NotImplementedException(); return (ITensorVar)null; }
            );

            return AddRecursive_(expr, seed, variable);
        }

        /// <summary> Add a recursive For to the loop. Returns the For created. </summary>
        public IFor AddRecursive_(ITensor expr, ITensor seed, ITensorVar variable)
        {
            if (seed == null)
                throw new ArgumentException($"{nameof(seed)} can't be null, use AddOutput instead", nameof(seed));
            SequencesLocked = true;

            var f = expr.Match(
                (Tensor<float> exprF) => (IFor)new Tensor<float>.For(this, Fors.Count, exprF, (Tensor<float>)seed, (Tensor<float>.Var)variable),
                (Tensor<int> exprI) => new Tensor<int>.For(this, Fors.Count, exprI, (Tensor<int>)seed, (Tensor<int>.Var)variable),
                null
            );
            Fors.Add(f);
            _variables.Add(f, variable);
            return f;
        }

        /// <summary>
        /// Add an output For to the loop.
        /// Returns the For created.
        /// </summary>
        public Tensor<T>.For AddOutput<T>(Tensor<T> expr)
        {
            SequencesLocked = true;

            var f = new Tensor<T>.For(this, Fors.Count, expr, null, null);
            Fors.Add(f);
            return f;
        }

        public IFor AddOutput_(ITensor expr)
        {
            SequencesLocked = true;

            var f = expr.Match(
                (Tensor<float> exprF) => (IFor)new Tensor<float>.For(this, Fors.Count, exprF, null, null),
                (Tensor<int> exprI) => new Tensor<int>.For(this, Fors.Count, exprI, null, null),
                null
            );
            Fors.Add(f);
            return f;
        }

        public bool IsVariable(ITensorVar expr) => _variables.ContainsValue(expr);

        internal Loop Patch(Patch substitutions)
        {
            Loop result;
            if (substitutions.TryGetValue(this, out result)) return result;

            // patch the length
            var patchLength = (Scalar<int>)Length.Patch(substitutions);

            // patch the sequences and outputInfo from the loop
            var patchSequences = Sequences.Patch(substitutions);
            var outputsInfo = Fors.Select(f => f.OutputInfo).ToArray();
            var patchOutputsInfo = outputsInfo.Patch(substitutions);

            // the expression uses the loop variables
            // we do a first patch to detect if we need to create a new loop
            var blankSubstitutions = new Patch(substitutions);
            var expressions = Fors.Select(f => f.Expression).ToArray();
            var patchExpressions = expressions.Patch(blankSubstitutions);

            if (patchLength == Length
                && patchOutputsInfo == outputsInfo
                && patchSequences == Sequences
                && patchExpressions == expressions)
            {
                // the expressions have been correctly patched
                foreach (var (expr, value) in (expressions, patchExpressions).Zip())
                    substitutions.Add_(expr, value);
                result = this;
            }
            else
            {
                // we need to create a new loop in particular we need to create new hidden variables for the loop
                var loopName = LoopName();
                // create the patched Loop
                result = new Loop(loopName, patchLength);

                // created variables are automatically added to `substitutions`
                var createdVariables = Variables.Select(v => _patchVar(v, substitutions)).ToList();

                for (int i = 0; i < Sequences.Count; ++i)
                {
                    var seq = patchSequences[i];
                    var patchVar = createdVariables[i];
                    patchVar.Match(
                        (Tensor<float>.Var varF) => (ITensorVar)result.AddSeq((Tensor<float>)seq, varF, SequenceAxes[i]),
                        (Tensor<int>.Var varI) => result.AddSeq((Tensor<int>)seq, varI, SequenceAxes[i]),
                        null
                    );
                }

                foreach (var @for in Fors)
                {
                   IFor patchFor;

                    // we need to patch the expression to use the new variables
                    var patchExpr = (ITensor)@for.Expression.Patch(substitutions);
                    if (@for.IsRecursive)
                    {
                        // the output has been patched already, it should be in substitutions.
                        var patchOutput = (ITensor)@for.OutputInfo.Patch(substitutions);
                        // we just put the patched version of the variable in substitutions
                        var patchVar = (ITensorVar)@for.RecursiveVariable.Patch(substitutions);

                        patchFor = result.AddRecursive_(patchExpr, patchOutput, patchVar);
                    }
                    else
                    {
                        patchFor = result.AddOutput_(patchExpr);
                    }

                    substitutions.Add_(@for, patchFor);
                }
            }
            substitutions.Add(this, result);
            return result;
        }

        private static void _patchFor<T>(Tensor<T>.For @for, Loop loop, Patch substitutions, List<IFor> patchFors)
        {
            // the output has been patched already, it should be in substitutions.
            var patchOutput = (Tensor<T>)@for.OutputInfo?.Patch(substitutions);
            // we just put the patched version of the variable in substitutions
            var patchVar = (Tensor<T>.Var)@for.RecursiveVariable?.Patch(substitutions);
            // we need to patch the expression to use the new variables
            var patchExpr = (Tensor<T>)@for.Expression.Patch(substitutions);

            var patchFor = new Tensor<T>.For(loop, @for.Index, patchExpr, patchOutput, patchVar);
            patchFors.Add(patchFor);
            substitutions.Add_(@for, patchFor);
        }

        private static ITensorVar _patchVar(ITensorVar v, Patch substitutions)
        {
            var lastChar = v.Name[v.Name.Length - 1];
            var endsWithDigit = lastChar >= '0' && lastChar <= '9';
            var name = v.Name;
            if (endsWithDigit)
                name = v.Name.Substring(0, v.Name.Length - 1) + (char)((lastChar - '0') + 1);
            else
                name += "_1";
            ITensorVar res;
            if (v is Tensor<float>.Var)
                res = new Tensor<float>.Var(v.Shape.Patch(substitutions), name);
            else if (v is Tensor<int>.Var)
                res = new Tensor<int>.Var(v.Shape.Patch(substitutions), name);
            else if (v is Tensor<double>.Var)
                res = new Tensor<double>.Var(v.Shape.Patch(substitutions), name);
            else
                throw new NotImplementedException();

            substitutions.Add_(v, res);
            return res;
        }

        private static ITensorVar _createVar(ITensor outputInfo, System.Reflection.ParameterInfo infos)
        {
            if (outputInfo == null)
                return null;

            var res = null
                ?? _tryCreateVar<float>(outputInfo, infos)
                ?? _tryCreateVar<int>(outputInfo, infos)
                ?? _tryCreateVar<double>(outputInfo, infos)
            ;

            if (res == null) throw new NotImplementedException();
            else return res;
        }

        private static ITensorVar _tryCreateVar<T>(ITensor outputInfo, System.Reflection.ParameterInfo infos) =>
            infos.ParameterType == typeof(Tensor<T>) ? new Tensor<T>.Var(outputInfo.Shape, infos.Name) : null;

        private static Tensor<T> _try<T>(ITensor tensor, Func<Tensor<T>, Tensor<T>> f) =>
            tensor is Tensor<T> ? f((Tensor<T>)tensor) : null;
    }
}
