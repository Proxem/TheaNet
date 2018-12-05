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

namespace Proxem.TheaNet.Binding
{
    /// <summary>
    /// This processor counts references on each expression.
    /// </summary>
    public class ReferenceCounter : IProcessor
    {
        //public static void Reference(IExpr target, Dictionary<IExpr, int> references)
        //{
        //    var counter = new ReferenceCounter(references);
        //    target.Process(counter);
        //}

        Dictionary<IExpr, int> references;
        Tuple<IVar, IExpr>[] bindings;

        public int this[IExpr e] => references.ContainsKey(e) ? references[e] : 0;

        public ReferenceCounter(Dictionary<IExpr, int> references = null, Tuple<IVar, IExpr>[] bindings = null)
        {
            this.references = references ?? new Dictionary<IExpr, int>();
            this.bindings = bindings;
        }

        /// <summary>
        /// Mark a given expression as needed.
        /// Will increment it's count and will process every inputs.
        /// </summary>
        /// <param name="e"></param>
        public void Reference(IExpr e)
        {
            Increment(e);
            e.Process(this);
        }

        public void Dereference(IExpr e)
        {
            e.Traverse(Decrement, mode: TraverseMode.STOP_ON_VISITED);
        }

        private void Increment(IExpr e)
        {
            int count;
            var exists = this.references.TryGetValue(e, out count);
            this.references[e] = count + 1;
            //Trace.WriteLine($"Incrementing {e} => {count + 1}");
            //if (e.ToString() == "x") Debugger.Break();
        }

        public Dictionary<IExpr, int> Content => references;

        public void Decrement(IExpr expr)
        {
            if (references.ContainsKey(expr))
            {
                references[expr] -= 1;
                if (references[expr] < 0)
                {
#if REF
                    Trace.WriteLine($"ReferenceCounter decremented too many times {expr}.");
#if DEBUG
                    throw new Exception($"ReferenceCounter decremented too many times {expr}.");
#endif
#endif
                }
            }
            else
            {
#if REF
                Trace.WriteLine("ReferenceCounter tried to decrement unknown: " + expr);
#if DEBUG
                throw new Exception("ReferenceCounter tried to decrement unknown: " + expr);
#endif
#endif
            }
        }

        public void AssertIsDone()
        {
            foreach (var kv in references)
                if (kv.Value != 0)
                {
#if REF
                    Trace.WriteLine($"ReferenceCounter still have {kv.Value} references on expr {kv.Key}");
#if DEBUG
                    throw new Exception($"ReferenceCounter still have {kv.Value} references on expr {kv.Key}");
#endif
#endif
            }
        }

        /// <summary>
        /// If the given target has already been processed, just increments it's count.
        /// Else also processes the inputs.
        /// </summary>
        /// <param name="target"></param>
        private void Process(IExpr target)
        {
            if (!references.ContainsKey(target))
            {
                Increment(target);
                target.Process(this);
            }
            else
                Increment(target);
        }

        private void Process(IEnumerable<IExpr> targets)
        {
            foreach (var target in targets)
            {
                Process(target);
            }
        }

        void IProcessor.ProcessList<T, U>(XList<T, U> target) => Process(target.Inputs);

        public void ProcessFor<T>(Tensor<T>.For target)
        {
            var loop = target.Loop;
            if (references.ContainsKey(loop.Fors[0].Expression)) return;       // already done

            Process(loop.Sequences);      // var xx = sequence[i]

            foreach(var rec in loop.RecursiveFors)
            {
                Process(rec.RecursiveVariable);
                Process(rec.OutputInfo);
                Process(rec.Expression);
            }

            // storage variables for expressions
            var length = loop.Length;
            for (int o = 0; o < loop.Fors.Count; o++)
            {
                var expr = loop.Fors[o].Expression;
                Process(expr);
                Process(length);          // new Array<float>(length, ....)
                Process(expr.Shape);      // new Array<float>(..., Shape)
            }

            Process(length);            // for (int i = 0; i < length; i++)
        }

        public void ProcessFunctionCall<T>(IExpr<T> target, string name, params object[] extras)
        {
            var args = target.Inputs;
            foreach (var arg in args)
            {
                Process(arg);
            }
            if (extras == null) return;
            foreach (var extra in extras)
            {
                if (extra is IExpr) throw new Exception("Extras argument can't contain any expression.");
                ProcessArg(extra);
            }
        }

        private void ProcessArg(object arg)
        {
            if (arg is IExpr)
            {
                Process(((IExpr)arg));
            }
            else if (arg is IEnumerable<IExpr>)
            {
                foreach (var subarg in (IEnumerable<IExpr>)arg)
                {
                    Process(subarg);
                }
            }
            else if (arg is NamedObject)
            {
                ProcessArg(((NamedObject)arg).Object);
            }
            else if (arg is Lambda)
            {
                // TODO: count references in lambda
            }
            //else throw new NotImplementedException(arg.GetType().GetName());
        }

        public void ProcessLiteral<T>(IExpr<T> target, T value)
        {
            //Console.WriteLine($"Processing {target}");
        }

        public void ProcessVar<T>(IVar<T> target)
        {
            if (bindings == null) return;
            foreach (var v in bindings)
            {
                if (v.Item1 == target)
                {
                    Process(v.Item2);
                    return;
                }
            }
        }

        public void ProcessShared<T>(IShared<T> target)
        {
            if (bindings == null) return;
            foreach (var v in bindings)
            {
                if (v.Item1 == target)
                {
                    Process(v.Item2);
                    return;
                }
            }
        }

        public void ProcessSlice(XSlice target)
        {
            Process(target.Inputs);
        }

        public void ProcessElementwise<T>(Tensor<T>.Elementwise target)
        {
            var mapping = target.Vars.Zip(target.Inputs, Tuple.Create<IVar, IExpr>);
            var bindings = this.bindings != null ? this.bindings.Concat(mapping).ToArray() : mapping.ToArray();
            target.Abstraction.Process(new ReferenceCounter(this.references, bindings));
        }
    }
}
