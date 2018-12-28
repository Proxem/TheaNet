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

using Proxem.TheaNet.Binding;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Collections;
using Proxem.NumNet;
using Proxem.TheaNet.Operators.Tensors;
using System.Diagnostics;

using static Proxem.TheaNet.LoopNamer;

namespace Proxem.TheaNet
{
    /// <summary>A collection of functions to create Tensors or create functions from Tensors</summary>
    public partial class Op
    {
        public static Operators.RandomFactory Random = new Operators.RandomFactory();

        public static Func<R> Function<R>( IExpr<R> output,
            OrderedDictionary updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(output, updates, givens, name: name);
        }

        public static Func<T1, T2> Function<T1, T2>(IVar<T1> input, IExpr<T2> output,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input, output, updates, givens, name: name);
        }

        public static Scalar<int> Size(Scalar<int>[] shape) => shape.Aggregate((Scalar<int>)1, (d, a) => d * a);

        public static FunctionBinder.ParamsFunction<X> FiniteDifference_<X>(IList<IVar> inputs, Scalar<X> output, Tensor<X>.Shared x, IDictionary givens = null)
        {
            var epsilon = Scalar<X>("epsilon");
            var i_ = Scalar<int>("i");
            var j_ = Scalar<int>("j");
            var k_ = Scalar<int>("k");
            var indexes = new Scalar<int>.Var[0];
            if (x.NDim == 1) indexes = new[] { i_ };
            if (x.NDim == 2) indexes = new[] { i_, j_ };
            if (x.NDim == 3) indexes = new[] { i_, j_, k_ };

            var inputSet = new List<IVar>();
            foreach (var i in inputs) inputSet.Add(i);
            foreach (var i in indexes) inputSet.Add(i);
            inputSet.Add(epsilon);

            var eps = Op.OneHot(x.Shape, indexes, epsilon);
            var out_m_eps = (Scalar<X>)output.Patch(new Patch { [x] = x - eps });
            var out_p_eps = (Scalar<X>)output.Patch(new Patch { [x] = x + eps });
            var delta = (out_p_eps - out_m_eps) / (Numeric<X>.Two * epsilon);

            return Function(inputSet, delta, givens: givens);
        }

        /// <summary>
        /// Creates a function that help checking the gradient backpropagated to a shared.
        /// The function created will expect one argument for each given input and one float for "epsilon".
        /// The function returns a gradient computed by finite difference and by the <paramref name="computed"/> expression.
        /// </summary>
        /// <remarks>
        /// "epsilon" is the length of the step used during finite difference.
        /// The gradient is checked every time in a different direction.
        /// </remarks>
        /// <typeparam name="X">float</typeparam>
        /// <param name="inputs">The inputs of the graph. The created function will expect a value for each of these inputs</param>
        /// <param name="output">The function to derive</param>
        /// <param name="x">The gradient of x will be checked</param>
        /// <param name="computed">The gradient to be checked. By default will use the "Grad" operator.</param>
        /// <param name="givens"></param>
        /// <returns>The test function</returns>
        public static FunctionBinder.ParamsFunction<X, X> RandomGradientCheck<X>(IList<IVar> inputs, Scalar<X> output, Tensor<X> x, Tensor<X> computed = null, IDictionary givens = null)
        {
            var epsilon = Scalar<X>("epsilon");
            var inputSet = new List<IVar>();
            foreach (var i in inputs) inputSet.Add(i);
            inputSet.Add(epsilon);

            var eps = Random.Uniform(-epsilon, epsilon, x.Shape); eps.Name = nameof(eps);
            var x_m_eps = x - eps;
            var x_p_eps = x + eps;
            if (x.Name != null)
            {
                x_m_eps.Name = x.Name + "_m_eps";
                x_p_eps.Name = x.Name + "_p_eps";
            }

            var out_m_eps = (Scalar<X>)output.Patch(new Patch {[x] = x_m_eps });
            out_m_eps.Name = output.Name + "_m_eps";

            var out_p_eps = (Scalar<X>)output.Patch(new Patch {[x] = x_p_eps });
            out_p_eps.Name = output.Name + "_p_eps";

            var finite = (out_p_eps - out_m_eps); finite.Name = nameof(finite);

            computed = computed ?? Grad(output, x);
            var backpropagated = Numeric<X>.Two * Sum(eps * computed);
            return Function(inputSet, output: (finite, backpropagated), givens: givens);
        }

        /// <summary>
        /// Creates a function that help checking the gradient backpropagated to a shared.
        /// The function created will expect one argument for each given input and one float for "epsilon".
        /// The function returns a gradient computed by finite difference and by the <paramref name="computed"/> expression.
        /// </summary>
        /// <typeparam name="X">float</typeparam>
        /// <param name="inputs">The inputs of the graph. The created function will expect a value for each of these inputs</param>
        /// <param name="output">The function to derive</param>
        /// <param name="x">The gradient of x will be checked</param>
        /// <param name="computed">The gradient to be checked. By default will use the "Grad" operator.</param>
        /// <param name="givens"></param>
        /// <returns>The test function</returns>
        public static FunctionBinder.ParamsFunction<X, X> RandomGradientCheck<X>(IList<IVar> inputs, Scalar<X> output, Scalar<X> x, Scalar<X> computed = null, IDictionary givens = null)
        {
            var eps = Scalar<X>("epsilon");
            var inputSet = new List<IVar>();
            foreach (var i in inputs) inputSet.Add(i);
            inputSet.Add(eps);

            var x_m_eps = x - eps;
            var x_p_eps = x + eps;
            if (x.Name != null)
            {
                x_m_eps.Name = x.Name + "_m_eps";
                x_p_eps.Name = x.Name + "_p_eps";
            }

            var out_m_eps = (Scalar<X>)output.Patch(new Patch { [x] = x_m_eps });
            var out_p_eps = (Scalar<X>)output.Patch(new Patch { [x] = x_p_eps });

            if(output.Name != null && out_m_eps != output)
            {
                out_m_eps.Name = output.Name + "_m_eps";
                out_p_eps.Name = output.Name + "_p_eps";
            }

            var finite = (out_p_eps - out_m_eps) / (Numeric<X>.Two * eps);
            if (finite.IsZero)
                Trace.WriteLine($"The given output {output} doesn't depend on {x}", "Warning");
            finite.Name = nameof(finite);

            computed = computed ?? Grad(output, x);
            return Function(inputSet, output: (finite, computed), givens: givens);
        }

        /// <summary>
        /// A specialized version of <see cref="RandomGradientCheck{X}(IList{IVar}, TheaNet.Scalar{X}, Tensor{X}.Shared, Tensor{X}, Tensor{X}, IDictionary)"/>
        /// with only one input.
        /// </summary>
        public static Func<Array<X>, X, Tuple<X, X>> RandomGradientCheck<X>(Tensor<X>.Var input, Scalar<X> output, Tensor<X>.Shared x, Tensor<X> computed = null, IDictionary givens = null)
        {
            var epsilon = Scalar<X>("epsilon");
            var eps = Random.Uniform(-epsilon, epsilon, x.Shape);

            var out_m_eps = (Scalar<X>)output.Patch(new Patch {[x] = x - eps });
            var out_p_eps = (Scalar<X>)output.Patch(new Patch {[x] = x + eps });
            var finite = (out_p_eps - out_m_eps);
            computed = computed ?? Grad(output, x);

            var backpropagated = Numeric<X>.Two * Sum(eps * computed);
            return FunctionBinder.Function(input1: input, input2: epsilon, output1: finite, output2: backpropagated, givens: givens);
        }

        [Obsolete]
        public static Func<T1, T2, T3> Function<T1, T2, T3>(IVar<T1> input1, IVar<T2> input2, IExpr<T3> output,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input1, input2, output, updates, givens, name: name);
        }

        public static Func<T1, T2, T3> Function<T1, T2, T3>((IVar<T1> x1, IVar<T2> x2) input, IExpr<T3> output,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input.x1, input.x2, output, updates, givens, name: name);
        }

        [Obsolete]
        public static Func<T1, T2, T3, T4> Function<T1, T2, T3, T4>(IVar<T1> input1, IVar<T2> input2, IVar<T3> input3, IExpr<T4> output,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input1, input2, input3, output, updates, givens, name: name);
        }

        public static Func<T1, T2, T3, T4> Function<T1, T2, T3, T4>((IVar<T1> x1, IVar<T2> x2, IVar<T3> x3) input, IExpr<T4> output,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input.x1, input.x2, input.x3, output, updates, givens, name: name);
        }

        [Obsolete]
        public static Func<T1, T2, T3, T4, T5> Function<T1, T2, T3, T4, T5>(IVar<T1> input1, IVar<T2> input2, IVar<T3> input3, IVar<T4> input4, IExpr<T5> output,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input1, input2, input3, input4, output, updates, givens, name: name);
        }

        public static Func<T1, T2, T3, T4, T5> Function<T1, T2, T3, T4, T5>((IVar<T1> x1, IVar<T2> x2, IVar<T3> x3, IVar<T4> x4) input, IExpr<T5> output,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input.x1, input.x2, input.x3, input.x4, output, updates, givens, name: name);
        }

        public static Func<T1, IList<T2>> Function<T1, T2>(IVar<T1> input, IEnumerable<IExpr<T2>> outputs,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input, outputs, updates, givens, name: name);
        }

        [Obsolete]
        public static Func<T1, Tuple<T2, T3>> Function<T1, T2, T3>(IVar<T1> input, IExpr<T2> output1, IExpr<T3> output2,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input, output1, output2, updates, givens, name: name);
        }

        public static Func<T1, Tuple<T2, T3>> Function<T1, T2, T3>(IVar<T1> input, (IExpr<T2> x1, IExpr<T3> x2) output,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input, output.x1, output.x2, updates, givens, name: name);
        }

        public static Func<IList<T1>, IList<T2>> Function<T1, T2>(IEnumerable<IVar<T1>> inputs, IEnumerable<IExpr<T2>> outputs,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(inputs, outputs, updates, givens, name: name);
        }

        [Obsolete]
        public static Func<IList<T1>, Tuple<T2, T3>> Function<T1, T2, T3>(IEnumerable<IVar<T1>> inputs, IExpr<T2> output1, IExpr<T3> output2,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(inputs, output1, output2, updates, givens, name: name);
        }

        public static Func<IList<T1>, Tuple<T2, T3>> Function<T1, T2, T3>(IEnumerable<IVar<T1>> inputs, (IExpr<T2> x1, IExpr<T3> x2) output,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(inputs, output.x1, output.x2, updates, givens, name: name);
        }

        public static Func<IList<T1>, T2> Function<T1, T2>(IEnumerable<IVar<T1>> inputs, IExpr<T2> output,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(inputs, output, updates, givens, name: name);
        }

        [Obsolete]
        public static Func<T1, T2, IList<T3>> Function<T1, T2, T3>(IVar<T1> input1, IVar<T2> input2, IList<IExpr<T3>> outputs,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input1, input2, outputs, updates, givens, name: name);
        }

        public static Func<T1, T2, IList<T3>> Function<T1, T2, T3>((IVar<T1> x1, IVar<T2> x2) input, IList<IExpr<T3>> outputs,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input.x1, input.x2, outputs, updates, givens, name: name);
        }

        [Obsolete]
        public static Func<T1, T2, Tuple<T3, T4>> Function<T1, T2, T3, T4>(IVar<T1> input1, IVar<T2> input2, IExpr<T3> output1, IExpr<T4> output2,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input1, input2, output1, output2, updates, givens, name: name);
        }

        public static Func<T1, T2, Tuple<T3, T4>> Function<T1, T2, T3, T4>((IVar<T1> x1, IVar<T2> x2) input, (IExpr<T3> x1, IExpr<T4> x2) output,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input.x1, input.x2, output.x1, output.x2, updates, givens, name: name);
        }

        public static Action Function(OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(updates, givens, name: name);
        }

        public static Action<T1> Function<T1>(IVar<T1> input,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input, updates, givens, name: name);
        }

        [Obsolete]
        public static Action<T1, T2> Function<T1, T2>(IVar<T1> input1, IVar<T2> input2,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input1, input2, updates, givens, name: name);
        }

        public static Action<T1, T2> Function<T1, T2>((IVar<T1> x1, IVar<T2> x2) input,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input.x1, input.x2, updates, givens, name: name);
        }

        [Obsolete]
        public static Action<T1, T2, T3> Function<T1, T2, T3>(IVar<T1> input1, IVar<T2> input2, IVar<T3> input3,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input1, input2, input3, updates, givens, name: name);
        }

        public static Action<T1, T2, T3> Function<T1, T2, T3>((IVar<T1> x1, IVar<T2> x2, IVar<T3> x3) input,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input.x1, input.x2, input.x3, updates, givens, name: name);
        }

        [Obsolete]
        public static Action<T1, T2, T3, T4> Function<T1, T2, T3, T4>(IVar<T1> input1, IVar<T2> input2, IVar<T3> input3, IVar<T4> input4,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input1, input2, input3, input4, updates, givens, name: name);
        }

        public static Action<T1, T2, T3, T4> Function<T1, T2, T3, T4>((IVar<T1> x1, IVar<T2> x2, IVar<T3> x3, IVar<T4> x4) input,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(input.x1, input.x2, input.x3, input.x4, updates, givens, name: name);
        }

        public static Action<IList<T1>> Function<T1>(IEnumerable<IVar<T1>> inputs,
            OrderedDictionary/*<TensorExpr.Shared, TensorExpr>*/ updates = null, IDictionary givens = null, string name = null)
        {
            return FunctionBinder.Function(inputs, updates, givens, name: name);
        }

        public static FunctionBinder.ParamsFunction<R> Function<R>(IEnumerable<IVar> inputs, IExpr<R> output, OrderedDictionary updates = null, IDictionary givens = null, string name = null)
            => FunctionBinder.Function_(inputs, output, updates, givens, name: name);

        [Obsolete]
        public static FunctionBinder.ParamsFunction<R1, R2> Function<R1, R2>(IEnumerable<IVar> inputs, IExpr<R1> output1, IExpr<R2> output2, OrderedDictionary updates = null, IDictionary givens = null, string name = null)
            => FunctionBinder.Function_(inputs, output1, output2, updates, givens, name: name);

        public static FunctionBinder.ParamsFunction<R1, R2> Function<R1, R2>(IEnumerable<IVar> inputs, (IExpr<R1> x1, IExpr<R2> x2) output, OrderedDictionary updates = null, IDictionary givens = null, string name = null)
            => FunctionBinder.Function_(inputs, output.x1, output.x2, updates, givens, name: name);

        /// <summary>
        /// Compile a funtion with the given signature.
        /// </summary>
        /// <typeparam name="FType">The signature of the returned function. Must be a delegate.</typeparam>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        /// <param name="updates"></param>
        /// <param name="givens"></param>
        /// <param name="useParams">The result function will accept a variable number of arguments as input.</param>
        public static FType Function<FType>(IList<IVar> inputs, IList<IExpr> outputs, OrderedDictionary updates = null, IDictionary givens = null, bool useParams = false, string name = null)
            where FType: class
            => FunctionBinder.Compile<FType>(inputs, outputs, updates, givens, useParams: useParams, name: name);

        /// <summary>
        /// From Theano's documentation
        /// "Scalar costs only can be directly handled by grad."
        /// "Arrays are handled through repeated applications."
        /// see also: http://deeplearning.net/software/theano/extending/op.html#grad
        /// </summary>
        public static Scalar<T> Grad<T>(Scalar<T> cost, Scalar<T> wrt)
        {
            var bp = cost.Backpropagation;
            if (bp == null)
            {
                cost.Backpropagation = bp = Backpropagation.Backward(cost, Numeric<T>.One);
            }
            if (bp.ScalarDerivatives.ContainsKey(wrt))
                return (Scalar<T>)bp.ScalarDerivatives[wrt];
            else
                return Numeric<T>.Zero;
        }

        public static Tensor<T> Grad<T>(Scalar<T> cost, Tensor<T> wrt)
        {
            var bp = cost.Backpropagation;
            if (bp == null)
            {
                cost.Backpropagation = bp = Backpropagation.Backward(cost, Numeric<T>.One);
            }
            var result = (Tensor<T>)bp.TensorDerivatives[wrt];
            wrt.AssertOfDim(result.NDim);
            return result;
        }

        public static IList<Tensor<T>> Grad<T>(Scalar<T> cost, Tensor<T>.Symbol[] wrt)
        {
            var bp = cost.Backpropagation;
            if (bp == null)
            {
                cost.Backpropagation = bp = Backpropagation.Backward(cost, Numeric<T>.One);
            }
            var result = new List<Tensor<T>>();
            foreach (var shared in wrt)
            {
                ITensor derivative;
                if (bp.TensorDerivatives.TryGetValue(shared, out derivative))
                {
                    ((ITensor<T>)derivative).AssertOfShape(shared);
                    result.Add((Tensor<T>)derivative);
                }
                else
                {
                    Trace.WriteLine($"No gradient for {shared.Name}", "Warning");
                    result.Add(ZerosLike(shared));
                }
            }
            return result;
        }

        public static Dictionary<Tensor<R>.Symbol, Tensor<R>> Grad<T, R>(Scalar<T> cost)
        {
            var bp = cost.Backpropagation;
            if (bp == null)
                cost.Backpropagation = bp = Backpropagation.Backward(cost, Numeric<T>.One);

            var result = new Dictionary<Tensor<R>.Symbol, Tensor<R>>();
            foreach (var shared in bp.TensorDerivatives.Keys)
                if (shared is Tensor<R>.Symbol)
                    result[(Tensor<R>.Symbol)shared] = (Tensor<R>)bp.TensorDerivatives[shared];

            return result;
        }

        public static Dictionary<Tensor<T>.Symbol, Tensor<T>> Grad<T>(Scalar<T> cost) => Grad<T, T>(cost);

        public static Tensor<T> Scan<T>(Func<Tensor<T>, Tensor<T>> fn, Tensor<T> sequence, int axis = 0, string name = null)
        {
            var loop = new Loop(name ?? LoopName(), fn, new[] { sequence }, null, axis);
            return (Tensor<T>)loop.Fors[0];
        }

        public static Tensor<T> Scan<T>(Func<Tensor<T>, Tensor<T>, Tensor<T>> fn, Tensor<T> sequence, Tensor<T> outputsInfo, int axis = 0, string name = null)
        {
            var loop = new Loop(name ?? LoopName(), fn, new[] { sequence }, outputsInfo == null ? null : new[] { outputsInfo }, axis);
            return (Tensor<T>)loop.Fors[0];
        }

        public static Tensor<T> Scan<T>(Func<Tensor<T>, Tensor<T>, Tensor<T>> fn, IList<Tensor<T>> sequences, Tensor<T> outputsInfo = null, int axis = 0, string name = null)
        {
            var loop = new Loop(name ?? LoopName(), fn, sequences.Cast<ITensor>().ToList(), outputsInfo == null ? null : new[] { outputsInfo }, axis);
            return (Tensor<T>)loop.Fors[0];
        }

        public static Tensor<T> Scan<T>(Func<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> fn, IList<Tensor<T>> sequences, Tensor<T> outputsInfo = null, int axis = 0, string name = null)
        {
            var loop = new Loop(name ?? LoopName(), fn, sequences.Cast<ITensor>().ToList(), outputsInfo == null ? null : new[] { outputsInfo }, axis);
            return (Tensor<T>)loop.Fors[0];
        }

        public static IList<Tensor<T>> Scan<T>(Func<Tensor<T>, Tensor<T>, IList<Tensor<T>>> fn, Tensor<T> sequences, Tensor<T>[] outputsInfo, int axis = 0, string name = null)
        {
            var loop = new Loop(name ?? LoopName(), fn, new[] { sequences }, outputsInfo, axis);
            return loop.Fors.Cast<Tensor<T>>().ToArray();
        }

        // 3 args many outputs
        //public static IList<Tensor<T>> Scan<T>(Func<Tensor<T>, Tensor<T>, Tensor<T>, IList<Tensor<T>>> fn, IList<Tensor<T>> sequences, Tensor<T>[] outputsInfo, int axis = 0)
        //{
        //    var loop = new Loop<T>(LoopName(), fn, sequences, outputsInfo, axis);
        //    return loop.Fors.ToArray();
        //}

        public static IList<Tensor<T>> Scan<T>(Func<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, IList<Tensor<T>>> fn, Tensor<T>[] sequences, Tensor<T>[] outputsInfo, int axis = 0, string name = null)
        {
            var loop = new Loop(name ?? LoopName(), fn, sequences, outputsInfo, axis);
            return loop.Fors.Cast<Tensor<T>>().ToArray();
        }

        public static IList<IFor> Scan_(Delegate fn, IReadOnlyList<ITensor> sequences, IReadOnlyList<ITensor> outputsInfo, int axis = 0, string name = null)
        {
            var loop = new Loop(name ?? LoopName(), fn, sequences, outputsInfo, axis);
            return loop.Fors.ToArray();
        }

        public static XSlice Slice(Scalar<int> start, Scalar<int> stop)
            => XSlicer.Range(start, stop);

        public static XSlice Slice(Scalar<int> start, Scalar<int> stop, int step)
            => XSlicer.Range(start, stop, step);
    }
}
