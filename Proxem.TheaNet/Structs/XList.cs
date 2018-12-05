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
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Proxem.NumNet;

namespace Proxem.TheaNet
{
    public interface IArray: IExpr
    {
        IEnumerable<IExpr> Values { get; }
    }

    /// <summary>
    /// An expression list that can also be seen as a list expression.
    /// Has implicit conversion with `Expr[]`.
    /// </summary>
    /// <typeparam name="T">The content of the array.</typeparam>
    /// <typeparam name="U">T is of the form `IExpr{ U }`</typeparam>
    /// <remarks>Examples:
    ///     `XList{ Scalar {X}, X}`
    ///     `XList{Tensor{X}, NumNet.Array{X}}`
    /// </remarks>
    public class XList<T, U> : Expr, IArray, IExpr<U[]>, IEnumerable<T>
        where T : class, IExpr<U>
    {
        IEnumerable<IExpr> IArray.Values => Inputs;
        public IReadOnlyList<T> Values => (IReadOnlyList<T>)Inputs;

        public XList(T[] values): base("List", values) { }

        public T this[int i] => (T)Inputs[i];

        public int Count => Inputs.Count;

        public static implicit operator XList<T, U>(T[] values) => new XList<T, U>(values);

        public static implicit operator T[] (XList<T, U> array) => (T[])array.Inputs;

        public override string ToString() => "[" + string.Join(", ", Inputs) + "]";

        public override IExpr Patch(Patch substitutions)
        {
            var newArray = new T[Count];
            bool same = true;
            for (int i = 0; i < Count; i++)
            {
                newArray[i] = (T)this.Inputs[i].Patch(substitutions);
                if (newArray[i] != this.Inputs[i]) same = false;
            }
            return same ? this : new XList<T, U>(newArray);
        }

        public override void Process(IProcessor processor) => processor.ProcessList(this);

        public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)Inputs).GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => Inputs.GetEnumerator();
    }

    public static class XListExtension
    {
        public static XList<Tensor<T>, Array<T>> ToStructArray<T>(this Tensor<T>[] values) => new XList<Tensor<T>, Array<T>>(values);
        public static XList<Scalar<T>, T> ToStructArray<T>(this Scalar<T>[] values) => new XList<Scalar<T>, T>(values);
    }
}
