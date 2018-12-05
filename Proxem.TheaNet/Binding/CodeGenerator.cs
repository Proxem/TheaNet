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

namespace Proxem.TheaNet.Binding
{
    /// <summary>
    /// Assigns C# variables to TheaNet expression.
    /// Handles buffers.
    /// </summary>
    public class CodeGenerator
    {
        public void Visit(IExpr expr, Compiler compiler)
        {
            if (expr is IConst)
                VisitConst(expr as IConst, compiler);
            else if (expr is IShared)
                VisitShared(expr as IShared, compiler);
            else if (expr is IVar)
                VisitVar(expr as IVar, compiler);
            else if (expr is XSlice)
                VisitSlice(expr as XSlice, compiler);
            else if (expr is IFor)
                VisitFor(expr as IFor, compiler);
            else if (expr is Tensor<float>.Elementwise)
                VisitElementwise(expr as Tensor<float>.Elementwise, compiler);
            else if (expr is Tensor<int>.Elementwise)
                VisitElementwise(expr as Tensor<int>.Elementwise, compiler);
            else if (expr is Tensor<double>.Elementwise)
                VisitElementwise(expr as Tensor<double>.Elementwise, compiler);
            else if (expr is IElementwise)
                throw new NotImplementedException("VisitElementwise for " + expr.GetType());
            else
                VisitNAry(expr, compiler);
        }

        /// <summary>
        /// A list of variables that are defined in the current scope.
        /// </summary>
        private readonly HashSet<IVar> loopVariables = new HashSet<IVar>();

        public virtual void VisitConst(IConst @const, Compiler compiler)
        {
            compiler.Scope.Declare(@const, compiler);
            compiler.CheckShape(@const, this);

            if (@const.Name != null)
            {
                var comment = GetComment(@const, compiler);
                compiler.EmitAssign(@const, @const.Literal, comment);
            }
        }

        public virtual void VisitFor(IFor @for, Compiler compiler)
        {
            compiler.Scope.Declare(@for, compiler);
            var loop = @for.Loop;
            if (loop.Compiler != compiler)
            {
                loop.Compiler = compiler;

                // Loop-invariant code motion: sequences variables are not yet declared, so expressions containing the latter will not be compiled, only constant expression will
                // https://en.wikipedia.org/wiki/Loop-invariant_code_motion
                using (LoopInvariant(loop.Variables))
                {
                    for (int o = 0; o < loop.Fors.Count; o++)
                    {
                        var expr = loop.Fors[o].Expression;
                        compiler.CompileExpr(expr, this);
                    }
                }

                // compile sequences
                foreach (var sequence in loop.Sequences)
                {
                    compiler.CompileExpr(sequence, this);
                }

                // compile seeds
                foreach (var outputInfo in loop.RecursiveFors.Select(f => f.OutputInfo))
                {
                    compiler.CompileExpr(outputInfo, this);
                    compiler.DecCount(outputInfo);
                }

                // declare storage variables for expressions
                var length = loop.Length;
                length.Comment = $"length of loop '{loop.Name}'";
                compiler.CompileExpr(length, this);

                foreach (var f in loop.Fors)
                {
                    var expr = f.Expression;
                    compiler.CompileExpr(expr.Shape, this);
                    compiler.Scope.Declare(f, compiler);

                    compiler.DecCount(length);
                    foreach (var axis in expr.Shape)
                        compiler.DecCount(axis);

                    string comment = null;
                    if (compiler.Verbose)
                    {
                        expr.Comment = expr.Comment ?? $"result for loop {f} = {expr}";
                        comment = compiler.RefComment(f, f.Shape, expr.Comment);
                    }

                    var buff = compiler.GetBuffer(f);
                    if (buff == null)
                    {
                        var innerType = ((ITensor)f).Match(
                            (Tensor<float> tf) => "float",
                            (Tensor<int> ti) => "int",
                            null
                        );
                        var shape = compiler.Scope.GetVar(length);
                        if (expr.NDim > 0)
                            shape += ", " + string.Join(", ", expr.Shape.Select(axis => compiler.Scope.GetVar(axis)));
                        compiler.EmitAssign(f, $"new Array<{innerType}>({shape})", comment);
                    }
                    else
                        compiler.EmitAssign(f, compiler.GetBufferName(buff), comment);
                }

                // create a scope for the variable private to the loop (recursive for example)
                compiler.EmitStartBlock(null, $"Start of {loop}.");

                // declare recursion variables (outputsInfo != null)
                foreach (var rec in loop.RecursiveFors)
                {
                    var outputInfo = rec.OutputInfo;
                    var variable = rec.RecursiveVariable;
                    compiler.Scope.Declare(variable, compiler, variable.Name);
                    compiler.EmitAssign(variable, compiler.Scope.GetVar(outputInfo), compiler.RefComment(variable, outputInfo));
                    compiler.DecCount(rec.RecursiveVariable);
                }

                // write the beginning of the `for` loop.
                compiler.DecCount(length);
                compiler.EmitStartBlock($"for (int i = 0; i < {compiler.Scope.GetVar(length)}; i++)", $"= ({compiler.GetCount(length)})");

                // extract items from sequences
                for (int i = 0; i < loop.Sequences.Count; i++)
                {
                    var seq = loop.Sequences[i];
                    var variable = loop.Variable(seq);
                    compiler.Scope.Declare(variable, compiler);

                    compiler.DecCount(seq);
                    var comment = compiler.RefComment(variable, seq);
                    var axis = loop.SequenceAxes[i];
                    if (axis == 0)
                        compiler.EmitAssign(variable, $"{compiler.Scope.GetVar(seq)}[i]", comment);
                    else
                        compiler.EmitAssign(variable, $"SliceAlong({compiler.Scope.GetVar(seq)}, {axis}, i)", comment);
                }

                // compile and store expressions
                for (int o = 0; o < loop.Fors.Count; o++)
                {
                    var expr = loop.Fors[o].Expression;
                    compiler.CompileExpr(expr, this);

                    compiler.DecCount(expr);
                    var comment = compiler.RefComment(loop.Fors[o], expr);
                    compiler.EmitStore(loop, o, expr, comment);
                }

                // copy recursion variable for next iteration
                foreach (var rec in loop.RecursiveFors)
                {
                    var variable = loop.Variable(rec);
                    compiler.DecCount(rec.Expression);
                    var comment = compiler.RefComment(variable, rec.Expression);
                    compiler.EmitAliasing(variable, rec.Expression, comment);
                }

                compiler.EmitEndBlock();
                compiler.CheckShape(@for, this);

#if REF
                // the recursive variable aren'tvisible outside the scope, so they shouldn't be referenced anymore
                foreach (var v in loop.Fors.Select(f => f.RecursiveVariable))
                    if (v != null && compiler.GetCount(v) != 0)
                        throw new Exception("Recursive variable not decremented");
#endif
                compiler.EmitEndBlock();
            }
        }

        public virtual void VisitShared(IShared shared, Compiler compiler)
        {
            compiler.Scope.Declare(shared, compiler);
            var comment = GetComment(shared, compiler);

            if (shared is ITensor tensor)
            {
                var buff = compiler.GetBuffer(tensor);
                if (buff != null && (!buff.IsShared || buff.Name != shared.Name))
                {
                    compiler.EmitAssign(shared, $"Copy({Compiler.GetShared(shared)}, result: {compiler.GetBufferName(buff)})" + compiler.OfShape(@shared, this), comment);
                    return;
                }
            }
            compiler.EmitAssign(shared, Compiler.GetShared(shared) + compiler.OfShape(@shared, this), comment);
        }

        public virtual void VisitVar(IVar var, Compiler compiler)
        {
            if (!loopVariables.Contains(var))
                throw new ArgumentException($"An input of the graph was not provided and not given a value: {var.Name}");
        }

        /// <summary>
        /// Generate code for an Elementwise.
        /// Will try to group cascading "Elementwise objects" into one.
        /// </summary>
        /// <returns>True if the input was actually an Elementwise operator.</returns>
        public virtual bool VisitElementwise<T>(Tensor<T>.Elementwise elementwise, Compiler compiler)
        {
            if (loopVariables.Count > 0)
            {
                foreach (var e in elementwise.Inputs)
                    compiler.CompileExpr(e, this);
                if (!elementwise.Inputs.All(e => compiler.Scope.Contains(e)))
                    return true;     // part of the expression was not reachable, exit (processed = true)
            }

            var arguments = elementwise.Vars.Zip(elementwise.Inputs, Tuple.Create).ToList();
            var expr = ApplyLambdaAbstraction(elementwise.Abstraction, arguments, compiler);

            using (LoopInvariant(arguments.Select(_ => _.Item1)))
                compiler.CompileExpr(expr, this);

            foreach (var t in arguments)
            {
                compiler.CompileExpr(t.Item2, this);
            }
            if (!arguments.All(t => compiler.Scope.Contains(t.Item2))) return true;   // at least one expression could not compile (possible during loop invariant code motion)

            foreach (var t in arguments)
                compiler.DecCount(t.Item2);

            string lambda;
            using (compiler.CreateScope())
            {
                // TODO: Is it necessary to create new variable here ? Is it just about renaming them ?
                var locals = arguments.Select(arg => new Scalar<float>.Var("_" + compiler.Scope.GetVar(arg.Item2))).ToList();
                foreach (var local in locals)
                    compiler.Scope.Declare(local, compiler);

                IExpr target = expr;
                var bindings = arguments.Zip(locals, (a, l) => Tuple.Create<IVar, IExpr>(a.Item1, l)).ToArray();
                foreach (var binding in bindings)
                {
                    if (binding.Item1 == target)
                    {
                        target = binding.Item2;
                        break;
                    }
                }
                var body = CodeProcessor.Process(compiler, target, bindings);

                var vars = locals.Count == 1 ?
                    locals[0].ToString() :
                    "(" + string.Join(", ", locals) + ")";
                lambda = vars + " => " + body;

                foreach (var t in arguments)
                {
                    compiler.DecCount(t.Item1);
#if REF
                    if (compiler.GetCount(t.Item1) != 0)
                        throw new Exception("Lambda variable not decremented");
#endif
                }
            }

            var args = string.Join(", ", arguments.Select(t => compiler.Scope.GetVar(t.Item2)));
            var code = $"NN.Apply({args}, {lambda})";

            var buff = compiler.GetBuffer(elementwise);
            if (buff != null)
                code = $"{code.Substring(0, code.Length - 1)}, result: {compiler.GetBufferName(buff)})";

            code += compiler.OfShape(elementwise, this);

#if SMART
            // TODO: when reference-counting works
            if (elementwise.Name == null && compiler.GetCount(elementwise) == 1)
            {
                compiler.Scope.Declare(elementwise, compiler, code);
            }
            else
#endif
            {
                compiler.Scope.Declare(elementwise, compiler);

                string comment = null;
                if (compiler.Verbose)
                {
                    elementwise.Comment = elementwise.Comment ?? InlineCodeGenerator.GetCode(elementwise);
                    comment = compiler.RefComment(elementwise, arguments.Select(arg => arg.Item2), elementwise.Comment);
                }

                compiler.EmitAssign(elementwise, code, comment);
            }
            return true;
        }

        /// <summary>
        /// Apply the lambda abstraction on the arguments and perform beta-reductions
        /// </summary>
        /// <param name="abstraction"></param>
        /// <param name="arguments"></param>
        /// <param name="compiler"></param>
        /// <returns></returns>
        private Scalar<T> ApplyLambdaAbstraction<T>(Scalar<T> abstraction, List<Tuple<Scalar<T>.Var, Tensor<T>>> arguments, Compiler compiler)
        {
            for (int i = 0; i < arguments.Count; i++)
            {
                var fill = arguments[i].Item2 as Operators.Tensors.Fill<float>;
                if (fill != null)
                {
                    compiler.CompileExpr(fill.x, this);
                    abstraction = (Scalar<T>)abstraction.Patch(new Patch { [arguments[i].Item1] = fill.x });
                    compiler.DecCount(arguments[i].Item1);
                    compiler.Dereference(arguments[i].Item2);
                    arguments.RemoveAt(i);
                    --i;
                    continue;
                }

                var elementwise = arguments[i].Item2 as Tensor<T>.Elementwise;
                if (elementwise != null && !compiler.Scope.Contains(elementwise) && compiler.GetCount(elementwise) <= 1)
                {
                    var elementwiseArguments = elementwise.Vars.Zip(elementwise.Inputs, (var, input) => Tuple.Create(var, input)).ToList();
                    var elementwiseAbstraction = ApplyLambdaAbstraction(elementwise.Abstraction, elementwiseArguments, compiler);

                    abstraction = BetaReduce(ref i, abstraction, arguments, elementwiseAbstraction, elementwiseArguments, compiler);
                    continue;
                }
            }
            return abstraction;
        }

        const int MaxLambdaVars = 4;

        private static Scalar<T> BetaReduce<T>(ref int i,
            Scalar<T> abstraction1, List<Tuple<Scalar<T>.Var, Tensor<T>>> arguments1,
            Scalar<T> abstraction2, List<Tuple<Scalar<T>.Var, Tensor<T>>> arguments2,
            Compiler compiler
        )
        {
            var predictedSize = arguments1.Count() - 1 + arguments2.Count(arg => arguments1.FindIndex(argument2 => argument2.Item2 == arg.Item2) == -1);
            if (predictedSize > MaxLambdaVars) return abstraction1;

            var reducedVariable = arguments1[i].Item1;
            compiler.DecCount(arguments1[i].Item1);
            compiler.DecCount(arguments1[i].Item2);
            arguments1.RemoveAt(i);
            --i;
            //for (int j = arguments2.Count - 1; j >= 0; j--)
            for (int j = 0; j < arguments2.Count; j++)
            {
                var pos = arguments1.FindIndex(argument1 => argument1.Item2 == arguments2[j].Item2);
                if (pos == -1)
                {
                    ++i;
                    arguments1.Insert(i, arguments2[j]);
                }
                else
                {
                    // alpha-conversion
                    compiler.DecCount(arguments2[j].Item1);
                    compiler.DecCount(arguments2[j].Item2);
                    abstraction2 = (Scalar<T>)abstraction2.Patch(new Patch { [arguments2[j].Item1] = arguments1[pos].Item1 });
                }
            }

            Debug.Assert(predictedSize == arguments1.Count);
            // beta-reduction
            return (Scalar<T>)abstraction1.Patch(new Patch { [reducedVariable] = abstraction2 });
        }

        public void VisitNAry(IExpr nary, Compiler compiler)
        {
            Debug.Assert(!compiler.Scope.Contains(nary));

            // asks for replacement candidates, particulary useful for x.Shape[0] when it can be simplified.
            var candidates = FindReplacementCandidates(compiler, nary);
            IExpr expr = null;
            foreach (var candidate in candidates)
            {
                if (compiler.Scope.Contains(candidate))
                {
                    expr = candidate;
                    break;
                }

                foreach (var e in candidate.Inputs)
                    compiler.CompileExpr(e, this);

                if (candidate.Inputs.All(t => compiler.Scope.Contains(t)))
                {
                    expr = candidate;
                    break;
                }

                // at least one expression could not compile (possible during loop invariant code motion)
                // let's try another candidate
                continue;
            }

            if (expr == null)
                // We didn't find a way to compile `nary`.
                // The caller will check success through `compiler.Scope.GetVar`,
                // and decide if it want to throw an exception
                return;

            if (!compiler.Scope.Contains(expr))
            {
                // TODO: here we decrement the 'expr' counter, but we incremented the 'nary' count
                foreach (var e in expr.Inputs)
                    compiler.DecCount(e);

#if SMART
                // TODO: when reference-counting works
                if (expr.Name == null && compiler.GetCount(expr) == 1)
                {
                    compiler.Scope.Declare(expr, compiler, code);
                }
                else
#endif
                {
                    compiler.Scope.Declare(expr, compiler);
                    var code = CodeProcessor.Process(compiler, expr) + compiler.OfShape(expr, this);
                    compiler.EmitAssign(expr, code, GetComment(expr, compiler));
                }
            }

            // `expr` have already been computed and stored in a var,
            // we also map `nary` to the same var
            if (expr != nary && !compiler.Scope.Contains(nary))
                compiler.Scope.Declare(nary, compiler, compiler.Scope.GetVar(expr));
        }

        /// <summary>
        /// Proposes one or more alternative to a given expressions.
        /// </summary>
        /// <remarks>For now only simplify Scalar&lt;int&gt;</remarks>
        /// <param name="compiler">compilation context</param>
        /// <param name="expr">expr to simplify</param>
        /// <returns>a list of candidates (often a singleton)</returns>
        public IEnumerable<IExpr> FindReplacementCandidates(Compiler compiler, IExpr expr)
        {
            // if the expr is already visible in the current scope or if it's a const, there is no need to replace it.
            bool isSimple = compiler.Scope.Contains(expr) || expr is IConst;
            if (isSimple)
                return new[] { expr };

            // candidates for replacements have been stored in the Equivalences dictionary (only Scalar<int> for now).
            var scalarInt = expr as Scalar<int>;
            bool hasRemplacements = scalarInt != null && ShapeExtension.Equivalences.ContainsKey(scalarInt);
            if (hasRemplacements)
            {
                // initialize candidates, takes all.
                var candidates = ShapeExtension.Equivalences[scalarInt].Cast<IExpr>().ToList();
                // try to find a candidate simple to compute
                var simple = candidates.FirstOrDefault(x => x is Scalar<int>.Const)
                    ?? candidates.FirstOrDefault(x => compiler.Scope.Contains(x))
                    ?? candidates.FirstOrDefault(x => x.Inputs.All(compiler.Scope.Contains))
                ;

                if (simple != null)
                    return new[] { simple };
                else
                    // No simple candidate, we order candidates by number of simple inputs
                    return candidates
                        .Where(x => x.FindAll<IVar>().All(compiler.Scope.Contains))
                        .OrderByDescending(x => x.Inputs.Count(compiler.Scope.Contains)).ToList();
            }
            else
                return new[] { expr };
        }

        public void VisitSlice(XSlice node, Compiler compiler)
        {
            Debug.Assert(!compiler.Scope.Contains(node));

            // default version
            foreach (var expr in node.Inputs)
            {
                compiler.CompileExpr(expr, this);
            }
            if (!node.Inputs.All(t => compiler.Scope.Contains(t))) return;   // at least one expression could not compile (possible during loop invariant code motion)
            foreach (var expr in node.Inputs)
            {
                compiler.DecCount(expr);
            }

            var code = CodeProcessor.Process(compiler, node);

            compiler.Scope.Declare(node, compiler);
            compiler.EmitAssign(node, code, GetComment(node, compiler));
        }

        private string GetComment(IExpr node, Compiler compiler)
        {
            if (compiler.Verbose)
            {
                node.Comment = node.Comment ?? node.ToString();
                return compiler.RefComment(node, node.Inputs, node.Comment);
            }
            // if non verbose only use user set Comment
            else
                return node.Comment;
        }

        /// <summary>
        /// Adds the given list of variables to the list of acceptable variables inside a `using` block.
        /// Usage:
        /// ~~~~~~~~~~~~ cs
        /// using(LoopInvariant(loopVariables))
        /// {
        ///    this. ...
        /// }
        /// ~~~~~~~~~~~~
        /// </summary>
        /// <param name="loopVariables">Variables to add</param>
        /// <returns>A disposabe object.</returns>
        private LoopInvariantCodeMover LoopInvariant(IEnumerable<IVar> loopVariables) => new LoopInvariantCodeMover(this, loopVariables);

        /// <summary>
        /// A disposable object that adds a list of variables to the current scope, and remove them when disposed.
        /// Use it through `codeGenerator.LoopInvariant`.
        /// </summary>
        private class LoopInvariantCodeMover : IDisposable
        {
            IEnumerable<IVar> loopVariables;
            private CodeGenerator parent;

            public LoopInvariantCodeMover(CodeGenerator parent, IEnumerable<IVar> loopVariables)
            {
                this.loopVariables = loopVariables;
                this.parent = parent;

                foreach (var v in loopVariables)
                    parent.loopVariables.Add(v);
            }

            public void Dispose()
            {
                foreach (var v in loopVariables)
                    parent.loopVariables.Remove(v);
            }
        }
    }
}