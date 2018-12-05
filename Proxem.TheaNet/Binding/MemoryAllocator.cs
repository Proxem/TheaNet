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
    /// Represents a memory zone and the Expression refering to it in a given context.
    /// </summary>
    public class BufferInfo
    {
        public readonly string Name;
        public readonly Scalar<int> Size;
        public readonly Scalar<int>[] Shape;
        public readonly Type Content;
        public readonly HashSet<IExpr> References = new HashSet<IExpr>();
        public readonly bool IsShared;
        public readonly bool IsResizable;

        private BufferInfo(string name, Scalar<int>[] shape, Type t, bool shared = false)
        {
            Name = name;
            Shape = shape;
            Size = Op.Size(shape);
            Content = t;
            IsShared = shared;
            IsResizable = !(Size is Scalar<int>.Const);
        }

        public static BufferInfo Allocate<T>(string name, Scalar<int>[] shape) => new BufferInfo(name, shape, typeof(T));

        public static BufferInfo UseShared<T>(IShared<T> shared) => new BufferInfo(shared.Name, (shared as ITensor).Shape, typeof(T), shared: true);
    }

    /// <summary>
    /// Creates a mapping between Expression and buffers.
    /// Traverses the computation graph in the same order than the compiler, in order to detect when buffer can be reused.
    /// </summary>
    public class MemoryAllocator: IProcessor
    {

        readonly Dictionary<IExpr, int> references;
        private readonly Tuple<IVar, IExpr>[] bindings;

        /// <param name="references">Created with ReferenceCounter</param>
        /// <param name="bindings"></param>
        public MemoryAllocator(Dictionary<IExpr, int> references, Tuple<IVar, IExpr>[] bindings = null)
        {
            this.bindings = bindings;
            this.references = references;
        }

        public List<BufferInfo> _assignables = new List<BufferInfo>();
        public Dictionary<ITensor, BufferInfo> Buffer = new Dictionary<ITensor, BufferInfo>();
        private HashSet<IExpr> processed = new HashSet<IExpr>();

        private string BufferName() => "_buffer" + _assignables.Count;

        private BufferInfo Allocate(ITensor tensor)
        {
            if (tensor is Tensor<float> tf)
                return Allocate(tf);
            else if (tensor is Tensor<int> ti)
                return Allocate(ti);
            else
                throw new NotImplementedException($"Memory allocation not supported for: {tensor.GetType().ToString()}.");
        }

        private BufferInfo Allocate<T>(Tensor<T> tensor)
        {
            var buff = BufferInfo.Allocate<T>(BufferName(), tensor.Shape);
            buff.References.Add(tensor);
            _assignables.Add(buff);
            return buff;
        }

        public BufferInfo FindAvailable(ITensor tensor)
        {
            if (tensor is Tensor<float> tf)
                return FindAvailable(tf);
            else if (tensor is Tensor<int> ti)
                return FindAvailable(ti);
            else
                throw new NotImplementedException($"Memory allocation not supported for: {tensor.GetType().ToString()}.");
        }

        public BufferInfo FindAvailable<T>(Tensor<T> tensor)
        {
            if (Buffer.ContainsKey(tensor)) return Buffer[tensor];

            var size = tensor.Size;
            //if (!(size is Scalar<int>.Const)) return null;

            return _assignables.FirstOrDefault(buff => CanAssign(buff, tensor, size)) ?? Allocate(tensor);
        }

        private int GetCount(IExpr e) => references.ContainsKey(e) ? references[e] : 0;
        public void Decrement(IExpr expr)
        {
            if (references.ContainsKey(expr))
            {
                references[expr] -= 1;
                if (references[expr] < 0)
                {
#if REF
                    Trace.WriteLine($"MemoryAllocator decremented too many times {expr}.");
#if DEBUG
                    throw new Exception($"MemoryAllocator decremented too many times {expr}.");
#endif
#endif
                }
            }
            else
            {
#if REF
                Trace.WriteLine("MemoryAllocator tried to decrement unknown: " + expr);
#if DEBUG
                throw new Exception("MemoryAllocator tried to decrement unknown: " + expr);
#endif
#endif
            }
        }

        public void Decrement(IEnumerable<IExpr> nodes)
        {
            foreach (var node in nodes) Decrement(node);
        }

        public void AssertIsDone()
        {
            foreach (var kv in references)
                if (kv.Value != 0)
                {
#if REF
                    Trace.WriteLine($"Memory allocator still have {kv.Value} references on expr {kv.Key}");
                    throw new Exception($"Memory allocator still have {kv.Value} references on expr {kv.Key}");
#endif
                }
        }

        private bool CanAssign<T>(BufferInfo buff, Tensor<T> tensor, Scalar<int> size)
        {
            // shared musn't be used to stock intermediary results
            if (buff.IsShared) return false;

            // return a buffer with the same size
            // (WillEqual is really cautious so we may allocate too many buffers)
            if (buff.Content == typeof(T) && buff.References.All(e => GetCount(e) == 0) && buff.Size.WillEqualTo(size))
            {
                if (!buff.Shape.WillEqualTo(tensor.Shape))
                {
                    return false;
                    // TODO: allow reshaping of avalaible buffer
                    // we need either to reset the shape at the end of the function
                    // or to reshape explictly every time a buffer is called
                    // or change buffer to float[] and create a new array with them each time
                }
                else
                {
                    return true;
                }
            }
            return false;
        }

        public void ProcessAndDec(IEnumerable<IExpr> nodes)
        {
            foreach (var node in nodes)
                Process(node);
            foreach (var node in nodes)
                Decrement(node);
        }

        public void ProcessAndDec(IExpr node)
        {
            Process(node);
            Decrement(node);
        }

        public void Process(IExpr node)
        {
            if (!processed.Contains(node))
            {
                processed.Add(node);
                node.Process(this);
            }
        }

        public void ProcessLiteral<T>(IExpr<T> target, T value) { }

        public void ProcessVar<T>(IVar<T> target) { }

        public void ProcessShared<T>(IShared<T> target)
        {
            if(target is ITensor tensor)
            {
                if (Buffer.ContainsKey(tensor))
                {
                    var buff = Allocate(tensor);
                    Buffer[tensor] = buff;
                }
                else
                {
                    var buff = BufferInfo.UseShared(target);
                    buff.References.Add(target);
                    Buffer[tensor] = buff;
                }
            }
        }

        void IProcessor.ProcessList<T, U>(XList<T, U> target) => ProcessAndDec(target.Inputs);

        public void ProcessFunctionCall<T>(IExpr<T> node, string name, object[] extras = null)
        {
            foreach (var expr in node.Inputs)
            {
                Process(expr);
            }
            Decrement(node.Inputs);

            if (node is ITensor tensor)
            {
                var buff = FindAvailable(tensor);

                if (name == "[]")
                {
                    var fullTensor = (ITensor)node.Inputs[0];
                    if (Buffer.ContainsKey(fullTensor))
                    {
                        buff = Buffer[fullTensor];
                        buff?.References.Add(node);
                    }
                    return;
                }

                Buffer[tensor] = buff;
                buff?.References.Add(node);
            }
        }

        public void ProcessTuple(ITuple target, params IExpr[] args)
        {
            // TODO allocate buffer to tuple of array (ITensorTuple)
            foreach (var expr in target.Inputs)
                Process(expr);
            Decrement(target.Inputs);
        }

        public void ProcessFor<T>(Tensor<T>.For @for)
        {
            var loop = @for.Loop;

            // compile sequences
            ProcessAndDec(loop.Sequences);
            ProcessAndDec(loop.Length);

            // decrement shapes
            foreach (var f in loop.Fors)
            {
                var buffFor = FindAvailable(f);
                Buffer[f] = buffFor;
                ProcessAndDec(f.Shape);
            }

            // find a buffer for the outputInfo
            foreach (var f in loop.Fors)
            {
                if (f.OutputInfo != null)
                {
                    Process(f.OutputInfo);
                    var buffOutput = FindAvailable(f.OutputInfo);
                    Buffer[f.OutputInfo] = buffOutput;
                    // prevent the buffer to be reused before the end of the first loop
                    buffOutput?.References.Add(f.RecursiveVariable);

                    Decrement(f.OutputInfo);
                }
            }

            foreach (var f in loop.Fors)
            {
                if (f.RecursiveVariable != null)
                {
                    // note: we find a buffer then we process
                    var buff = FindAvailable(f.Expression);
                    Buffer[f.Expression] = buff;
                    Buffer[f.RecursiveVariable] = buff;
                    buff?.References.Add(f.Expression);
                    buff?.References.Add(f.RecursiveVariable);
                }

                Process(f.Expression);
                processed.Add(f);
            }

            // we don't need the recursive variable anymore
            foreach (var f in loop.Fors)
            {
                // during reference count, recursive expressions are incremented twice,
                // once for the assignement to the result sequence
                if (f.OutputInfo != null)
                    Decrement(f.Expression);

                // once for the assignement to the recursive variable
                if (f.RecursiveVariable != null)
                    Decrement(f.RecursiveVariable);
                Decrement(f.Expression);
            }

            // all variable of the loop shouldn't have any references left when we scope out of the loop
            foreach (var f in loop.Fors)
            {
                int n = references[f.Expression];
                int m = f.RecursiveVariable != null ? references[f.RecursiveVariable] : 0;
                Debug.Assert(n == 0);
                Debug.Assert(m == 0);
            }
        }

        public void ProcessSlice(XSlice target) => ProcessAndDec(target.Inputs);

        public void ProcessElementwise<T>(Tensor<T>.Elementwise node)
        {
            foreach (var expr in node.Inputs)
                Process(expr);

            Decrement(node.Inputs);

            var mapping = node.Vars.Zip(node.Inputs, Tuple.Create<IVar, IExpr>);
            var bindings = this.bindings != null ? this.bindings.Concat(mapping).ToArray() : mapping.ToArray();
            node.Abstraction.Process(new MemoryAllocator(this.references, bindings));

            var buff = FindAvailable(node);
            Buffer[node] = buff;

            buff?.References.Add(node);
        }
    }
}
