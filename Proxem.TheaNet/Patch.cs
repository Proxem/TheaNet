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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Collections.Specialized;

namespace Proxem.TheaNet
{
    public class Patch
    {
        Dictionary<object, object> _substitutions = new Dictionary<object, object>();

        public readonly bool PreserveShape = true;

        public Patch(bool preserveShape = true)
        {
            PreserveShape = preserveShape;
        }

        public Patch(Patch substitutions) : this(substitutions._substitutions, substitutions.PreserveShape) {}

        public Patch(IDictionary givens, bool preserveShape = true):
            this(preserveShape)
        {
            if (givens != null)
            {
                foreach (DictionaryEntry e in givens)
                {
                    _substitutions[e.Key] = e.Value;
                }
            }
        }

        public bool ContainsKey(IExpr e) => _substitutions.ContainsKey(e);
        public bool ContainsValue(IExpr e) => _substitutions.ContainsValue(e);

        public IExpr<T> Process<T>(IExpr<T> expr)
        {
            return (IExpr<T>)expr.Patch(this);
        }

        public IExpr Process(IExpr expr) => expr.Patch(this);

        public void Process(OrderedDictionary updates)
        {
            if (updates == null) return;
            for (int i = 0; i < updates.Count; i++)
            {
                updates[i] = ((IExpr)updates[i]).Patch(this);
            }
        }

        public bool TryGetValue<T>(T expr, out T result)
            where T : class
        {
            result = null;
            object v;
            if (!_substitutions.TryGetValue(expr, out v)) return false;
            if (!typeof(T).IsAssignableFrom(v.GetType()))
            {
                // Is there an implicit conversion operator ?
                var types = new[] { v.GetType() };
                var methodInfo = typeof(T).GetMethod("op_Implicit",
                    BindingFlags.Public | BindingFlags.Static, null, types, null);
                if (methodInfo == null)
                    methodInfo = v.GetType().GetMethod("op_Implicit",
                        BindingFlags.Public | BindingFlags.Static, null, types, null);
                if (methodInfo == null)
                    throw new InvalidCastException($"Cannot cast given '{expr}' of type '{v.GetType().GetName()}' to expected type '{typeof(T).GetName()}'");
                v = methodInfo.Invoke(null, new[] { v });
            }
            result = (T)v;
            return true;
        }

        public void Add<T>(IExpr<T> expr, IExpr<T> value)
        {
            if(expr is ITensor tensor && PreserveShape)
                if(!ShapeExtension.CanEqualTo((value as ITensor).Shape, tensor.Shape))
                    throw new ArgumentException($"Can't patch {expr} with {value}");
            _substitutions[expr] = value;
        }

        public void Add(Loop expr, Loop value)
        {
            _substitutions[expr] = value;
        }

        public void Add_(IExpr expr, IExpr value)
        {
            if (expr is ITensor tensor && PreserveShape)
                if (!ShapeExtension.CanEqualTo((value as ITensor).Shape, tensor.Shape))
                    throw new ArgumentException($"Can't patch {expr} with {value}");
            _substitutions[expr] = value;
        }

        public object this[object expr]
        {
            set { _substitutions[expr] = value; }
        }
    }
}
