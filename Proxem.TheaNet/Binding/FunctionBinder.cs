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
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using Proxem.NumNet;

namespace Proxem.TheaNet.Binding
{
    public class FunctionBinder
    {
        #region One output

        public static Func<R> Function<R>(IExpr<R> output, OrderedDictionary updates = null, IDictionary givens = null, string name = "Function") =>
            Compile<Func<R>>(
                EmptyArray<Tuple<IVar, Type>>.Value,
                new[] { Tuple.Create((IExpr)output, typeof(R)) },
                updates,
                givens,
                name: name
            );

        public static Func<T, R> Function<T, R>(IVar<T> input, IExpr<R> output,
            OrderedDictionary/*<IExpr<T>.Shared, IExpr<T>>*/ updates = null,
            IDictionary givens = null,
            string name = null)
        {
            return Compile<Func<T, R>>(
                new[] { Tuple.Create((IVar)input, typeof(T)) },
                new[] { Tuple.Create((IExpr)output, typeof(R)) },
                updates,
                givens,
                name: name
            );
        }

        public static Func<T1, T2, R> Function<T1, T2, R>(IVar<T1> input1, IVar<T2> input2, IExpr<R> output,
            OrderedDictionary/*<IExpr<T>.Shared, IExpr<T>>*/ updates = null,
            IDictionary givens = null,
            string name = null)
        {
            return Compile<Func<T1, T2, R>>(
                new[] { Tuple.Create((IVar)input1, typeof(T1)), Tuple.Create((IVar)input2, typeof(T2)) },
                new[] { Tuple.Create((IExpr)output, typeof(R)) },
                updates,
                givens,
                name : name
            );
        }

        public static Func<T1, T2, T3, R> Function<T1, T2, T3, R>(IVar<T1> input1, IVar<T2> input2, IVar<T3> input3, IExpr<R> output,
            OrderedDictionary/*<IExpr<T>.Shared, IExpr<T>>*/ updates = null,
            IDictionary givens = null,
            string name = null
        ) =>
            Compile<Func<T1, T2, T3, R>>(
                new[] { Tuple.Create((IVar)input1, typeof(T1)), Tuple.Create((IVar)input2, typeof(T2)), Tuple.Create((IVar)input3, typeof(T3)) },
                new[] { Tuple.Create((IExpr)output, typeof(R)) },
                updates,
                givens,
                name : name
            );

        public static Func<T1, T2, T3, T4, R> Function<T1, T2, T3, T4, R>(IVar<T1> input1, IVar<T2> input2, IVar<T3> input3, IVar<T4> input4, IExpr<R> output,
            OrderedDictionary/*<IExpr<T>.Shared, IExpr<T>>*/ updates = null,
            IDictionary givens = null,
            string name = null
        ) =>
            Compile<Func<T1, T2, T3, T4, R>>(
                new[] { Tuple.Create((IVar)input1, typeof(T1)), Tuple.Create((IVar)input2, typeof(T2)), Tuple.Create((IVar)input3, typeof(T3)), Tuple.Create((IVar)input4, typeof(T4)) },
                new[] { Tuple.Create((IExpr)output, typeof(R)) },
                updates,
                givens,
                name: name
            );

        public static Func<IList<T>, R> Function<T, R>(
            IEnumerable<IVar<T>> inputs, IExpr<R> output,
            OrderedDictionary updates = null,
            IDictionary givens = null,
            string name = null)
        {
            var t = typeof(T);
            return Compile<Func<IList<T>, R>>(
                inputs.Select(i => Tuple.Create((IVar)i, t)).ToList(),
                new[] { Tuple.Create((IExpr)output, typeof(R)) },
                updates,
                givens,
                inputsInAList: t,
                name: name
            );
        }

        public delegate R ParamsFunction<R>(params object[] args);

        /// <summary>
        /// Creates a function with one output.
        /// The number of inputs and their types are determined during compilation.
        /// At runtime the function will fail if it don't receives as many inputs than expected.
        /// The inputs will be downcasted to the expected type inside the function.
        /// </summary>
        public static ParamsFunction<R> Function_<R>(
            IEnumerable<IVar> inputs,
            IExpr<R> output,
            OrderedDictionary updates = null,
            IDictionary givens = null,
            string name = null
        ) =>
            Compile<ParamsFunction<R>>(
                AddTypes(inputs),
                new[] { Tuple.Create((IExpr)output, typeof(R)) },
                updates,
                givens,
                useParams: true,
                name: name
            );

        #endregion
        #region Two outputs

        public static Func<T, Tuple<R1, R2>> Function<T, R1, R2>(
            IVar<T> input, IExpr<R1> output1, IExpr<R2> output2,
            OrderedDictionary updates = null,
            IDictionary givens = null,
            string name = null
        ) =>
            Compile<Func<T, Tuple<R1, R2>>>(
                new[] { Tuple.Create((IVar)input, typeof(T)) },
                new[] { Tuple.Create((IExpr)output1, typeof(R1)), Tuple.Create((IExpr)output2, typeof(R2)) },
                updates,
                givens,
                name: name
            );

        public static Func<T1, T2, Tuple<R1, R2>> Function<T1, T2, R1, R2>(
            IVar<T1> input1, IVar<T2> input2, IExpr<R1> output1, IExpr<R2> output2,
            OrderedDictionary updates = null,
            IDictionary givens = null,
            string name = null
        ) =>
            Compile<Func<T1, T2, Tuple<R1, R2>>>(
                new[] { Tuple.Create((IVar)input1, typeof(T1)), Tuple.Create((IVar)input2, typeof(T2)) },
                new[] { Tuple.Create((IExpr)output1, typeof(R1)), Tuple.Create((IExpr)output2, typeof(R2)) },
                updates,
                givens,
                name: name
            );

        public static Func<IList<T>, Tuple<R1, R2>> Function<T, R1, R2>(IEnumerable<IVar<T>> inputs, IExpr<R1> output1, IExpr<R2> output2,
            OrderedDictionary/*<IExpr<T>.Shared, IExpr<T>>*/ updates = null,
            IDictionary givens = null,
            string name = null)
        {
            var t = typeof(T);
            return Compile<Func<IList<T>, Tuple<R1, R2>>>(
                inputs.Select(i => Tuple.Create((IVar)i, t)).ToList(),
                new[] { Tuple.Create((IExpr)output1, typeof(R1)), Tuple.Create((IExpr)output2, typeof(R2)) },
                updates,
                givens,
                name: name
            );
        }

        public delegate Tuple<R1, R2> ParamsFunction<R1, R2>(params object[] args);

        /// <summary>
        /// Creates a function with two outputs.
        /// The number of inputs and their types are determined during compilation.
        /// At runtime the function will fail if it don't receives as many inputs than expected.
        /// The inputs will be downcasted to the expected type inside the function.
        /// </summary>
        public static ParamsFunction<R1, R2> Function_<R1, R2>(
            IEnumerable<IVar> inputs,
            IExpr<R1> output1,
            IExpr<R2> output2,
            OrderedDictionary updates = null,
            IDictionary givens = null,
            string name = null
        ) =>
            Compile<ParamsFunction<R1, R2>>(
                AddTypes(inputs),
                new[] { Tuple.Create((IExpr)output1, typeof(R1)), Tuple.Create((IExpr)output2, typeof(R2)) },
                updates,
                givens,
                useParams: true,
                name: name
            );

        #endregion
        #region Outputs in a list

        public static Func<T, IList<R>> Function<T, R>(
            IVar<T> input, IEnumerable<IExpr<R>> outputs,
            OrderedDictionary updates = null,
            IDictionary givens = null,
            string name = null
        ){
            var r = typeof(R);
            return Compile<Func<T, IList<R>>>(
                new[] { Tuple.Create((IVar)input, typeof(T)) },
                outputs.Select(o => Tuple.Create((IExpr)o, r)).ToList(),
                updates,
                givens,
                outputsInAList: r,
                name: name
            );
        }

        public static Func<T1, T2, IList<R>> Function<T1, T2, R>(
            IVar<T1> input1, IVar<T2> input2, IEnumerable<IExpr<R>> outputs,
            OrderedDictionary updates = null,
            IDictionary givens = null,
            string name = null
        ){
            var r = typeof(R);
            return Compile<Func<T1, T2, IList<R>>>(
                new[] { Tuple.Create((IVar)input1, typeof(T1)), Tuple.Create((IVar)input2, typeof(T2)) },
                outputs.Select(o => Tuple.Create((IExpr)o, r)).ToList(),
                updates,
                givens,
                outputsInAList: r,
                name: name
            );
        }

        public static Func<IList<T>, IList<R>> Function<T, R>(
            IEnumerable<IVar<T>> inputs, IEnumerable<IExpr<R>> outputs,
            OrderedDictionary updates = null,
            IDictionary givens = null,
            string name = null
        ) {
            var t = typeof(T);
            var r = typeof(R);
            return Compile<Func<IList<T>, IList<R>>>(
                inputs.Select(i => Tuple.Create((IVar)i, t)).ToList(),
                outputs.Select(o => Tuple.Create((IExpr)o, r)).ToList(),
                updates,
                givens,
                inputsInAList: t,
                outputsInAList: r,
                name: name
            );
        }

        #endregion
        #region No outputs
        public static Action Function(OrderedDictionary updates, IDictionary givens = null, string name = null) =>
            Compile<Action>(
                EmptyArray<Tuple<IVar, Type>>.Value,
                EmptyArray<Tuple<IExpr, Type>>.Value,
                updates,
                givens,
                name: name
            );

        public static Action<T1> Function<T1>(
            IVar<T1> input,
            OrderedDictionary updates,
            IDictionary givens = null,
            string name = null
        ) =>
            Compile<Action<T1>>(
                new[] { Tuple.Create((IVar)input, typeof(T1)) },
                EmptyArray<Tuple<IExpr, Type>>.Value,
                updates,
                givens,
                name: name
            );

        public static Action<T1, T2> Function<T1, T2>(
            IVar<T1> input1, IVar<T2> input2,
            OrderedDictionary updates,
            IDictionary givens = null,
            string name = null
        ) =>
           Compile<Action<T1, T2>>(
                new[] { Tuple.Create((IVar)input1, typeof(T1)), Tuple.Create((IVar)input2, typeof(T2)) },
                EmptyArray<Tuple<IExpr, Type>>.Value,
                updates,
                givens,
                name: name
            );

        public static Action<T1, T2, T3> Function<T1, T2, T3>(
            IVar<T1> input1, IVar<T2> input2, IVar<T3> input3,
            OrderedDictionary updates,
            IDictionary givens = null,
            string name = null
        ) =>
            Compile<Action<T1, T2, T3>>(
                new[] { Tuple.Create((IVar)input1, typeof(T1)), Tuple.Create((IVar)input2, typeof(T2)), Tuple.Create((IVar)input3, typeof(T3)) },
                EmptyArray<Tuple<IExpr, Type>>.Value,
                updates,
                givens,
                name: name
            );

        public static Action<T1, T2, T3, T4> Function<T1, T2, T3, T4>(
            IVar<T1> input1, IVar<T2> input2, IVar<T3> input3, IVar<T4> input4,
            OrderedDictionary updates,
            IDictionary givens = null,
            string name = null
        ) =>
            Compile<Action<T1, T2, T3, T4>>(
                new[] { Tuple.Create((IVar)input1, typeof(T1)), Tuple.Create((IVar)input2, typeof(T2)), Tuple.Create((IVar)input3, typeof(T3)), Tuple.Create((IVar)input4, typeof(T4)) },
                EmptyArray<Tuple<IExpr, Type>>.Value,
                updates,
                givens,
                name: name
            );

        public static Action<IList<T>> Function<T>(
            IEnumerable<IVar<T>> inputs,
            OrderedDictionary updates,
            IDictionary givens = null,
            string name = null
        ){
            var t = typeof(T);
            return Compile<Action<IList<T>>>(
                inputs.Select(i => Tuple.Create((IVar)i, t)).ToList(),
                EmptyArray<Tuple<IExpr, Type>>.Value,
                updates,
                givens,
                name: name
            );
        }
        #endregion

        /// <summary>
        /// The compiler used by the last call to Function.
        /// </summary>
        [ThreadStatic]
        // HACK: this Compiler should be passed as argument to the Compile function instead of being saved statically
        public static Compiler Compiler;

        public static Func<Compiler> CompilerFactory = () => new Compiler();

        public static FType Compile<FType>(IList<IVar> inputs, IList<IExpr> outputs,
            OrderedDictionary updates = null,
            IDictionary givens = null,
            Type inputsInAList = null,
            Type outputsInAList = null,
            bool useParams = false,
            string name = null
        )
            where FType : class =>
            Compile<FType>(
                inputs.Select(v => Tuple.Create(v, ExtractBaseType(v))).ToList(),
                outputs.Select(e => Tuple.Create(e, ExtractBaseType(e))).ToList(),
                updates, givens,
                useParams: useParams,
                name: name
            );

        private static Type ExtractBaseType(IExpr expr)
        {
            var exprType = expr.GetType();
            var ancestor = exprType.GetInterface("IExpr`1");
            var generics = ancestor?.GetGenericArguments();
            if (generics == null)
                throw new Exception($"No IExpr<?> ancestor was found for type {exprType}");
            if (generics.Length != 1)
                throw new Exception($"Was looking for a IExpr<?> ancestor for type {exprType}, but found IExpr<{string.Join<Type>(", ", generics)}>.");
            return generics[0];
        }

        /// <summary>
        /// Compiles a TheaNet graph to a function.
        /// </summary>
        /// <typeparam name="FType">expected function type</typeparam>
        /// <param name="inputs">inputs of the graph with their types</param>
        /// <param name="outputs">outputs of the graph with their types</param>
        /// <param name="updates">additional actions to performe in the function</param>
        /// <param name="givens">expressions to replace before compiling</param>
        /// <param name="inputsInAList">if true the function will expect arguments in a list</param>
        /// <param name="outputsInAList">if true the function will return results in a list</param>
        /// <param name="useParams">if true the function will expect arguments in a @param array</param>
        /// <param name="name"></param>
        /// <returns>A delegate of the required type</returns>
        public static FType Compile<FType>(IList<Tuple<IVar, Type>> inputs, IList<Tuple<IExpr, Type>> outputs,
            OrderedDictionary updates = null,
            IDictionary givens = null,
            Type inputsInAList = null,
            Type outputsInAList = null,
            bool useParams = false,
            string name = null
        )
             where FType: class
        {
            if (name == null) name = "Function";
            var compiler = CompilerFactory();
            compiler.EmitHeader();
            compiler.EmitStartBlock("public class DynClass: Proxem.TheaNet.Runtime");

            var inputSignature = InputSignature(inputs, inputsInAList, useParams);
            var outputSignature = OutputSignature(outputs, outputsInAList);

            compiler.EmitStartBlock($"public {outputSignature} {name}({inputSignature})");

            IList<IExpr> outs;
            if (givens != null)
            {
                var patch = new Patch(givens, preserveShape: false);
                outs = outputs.Select(output => patch.Process(output.Item1)).ToList();
                patch.Process(updates);
            }
            else
                outs = outputs.Select(_ => _.Item1).ToList();

            compiler.Reference(outs);
            compiler.Reference(updates);

            if (useParams)
                compiler.CompileArgs(inputs, null, "args");
            else
                compiler.CompileArgs(inputs, inputsInAList != null ? "args" : null, null);

            compiler.CreateMemoryMapping(outs, updates);

            var generator = new CodeGenerator();
            compiler.CompileExpr(outs, generator);
            compiler.EmitUpdates(updates, generator);

            compiler.EmitReturn(outs, outputsInAList);

            compiler.EmitEndBlock();

            compiler.EmitBuffers();

            compiler.EmitEndBlock();

            compiler.references.AssertIsDone();

            Compiler = compiler;
            return compiler.GetMethod<FType>(name);
        }

        private static string InputSignature(IList<Tuple<IVar, Type>> inputs, Type inputsInAList, bool useParams)
        {
            if (inputsInAList == null && !useParams)
                return string.Join(", ", inputs.Select(i => i.Item2.GetName() + " " + i.Item1.Name));
            else if (inputsInAList == null && useParams)
                return "params object[] args";
            else if (inputsInAList != null && useParams)
                return $"params {inputsInAList.GetName()}[] args";
            else
                return $"IList<{inputsInAList.GetName()}> args";
        }

        private static string OutputSignature(IList<Tuple<IExpr, Type>> outputs, Type outputsInAList)
        {
            if (outputsInAList == null && outputs.Count == 0)
                return "void";
            else if (outputsInAList == null && outputs.Count == 1)
                return outputs[0].Item2.GetName();
            else if (outputsInAList == null && outputs.Count > 1)
                return $"Tuple<{string.Join(", ", outputs.Select(i => i.Item2.GetName()))}>";
            else
                return $"IList<{outputsInAList.GetName()}>";
        }

        private static Type RuntimeType<T>(IExpr<T> expr) => typeof(T);

        private static IList<Tuple<T, Type>> AddTypes<T>(IEnumerable<T> exprs) where T : IExpr
        {
            var res = new List<Tuple<T, Type>>();
            foreach (T x in exprs)
            {
                dynamic y = x;
                Type t = RuntimeType(y);
                res.Add(Tuple.Create(x, t));
            }
            return res;
        }
    }
}
