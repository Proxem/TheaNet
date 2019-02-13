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
using Proxem.TheaNet;

namespace Proxem.TheaNet.Test
{
    /// <summary>
    /// The EqualityComparer recusrively compares two expressions,
    /// to determine their equality.
    /// </summary>
    public class EqualityComparer
    {
        public void Visit(IExpr expr, Tensor<float> compiler)
        {
            switch (expr)
            {
                case IConst c:
                    VisitConst(c, compiler);
                    break;
                case IShared s:
                    VisitShared(s, compiler);
                    break;
                case IVar v:
                    VisitVar(v, compiler);
                    break;
                case XSlice x:
                    VisitSlice(x, compiler);
                    break;
                case IFor f:
                    VisitFor(f, compiler);
                    break;
                default:
                    VisitNAry(expr, compiler);
                    break;
            }
        }

        public void Visit(IExpr expr, Scalar<float> compiler)
        {
            switch (expr)
            {
                case IConst c:
                    VisitConst(c, compiler);
                    break;
                case IShared s:
                    VisitShared(s, compiler);
                    break;
                case IVar v:
                    VisitVar(v, compiler);
                    break;
                case XSlice x:
                    VisitSlice(x, compiler);
                    break;
                case IFor f:
                    VisitFor(f, compiler);
                    break;
                default:
                    VisitNAry(expr, compiler);
                    break;
            }
        }

        //public readonly Tensor<float> Value;
        private bool areEqual = true;
        public bool AreEqual => areEqual;

        public void Arent()
        {
            areEqual = false;
        }

        /// <returns>True if one of the two object was able to be cast to the given type</returns>
        public bool VisitFloat(IExpr thiz, IExpr that)
        {
            var thizT = thiz as Scalar<float>;
            var thatT = that as Scalar<float>;
            if (thizT == null && thatT == null)
                return false;
            else if (thizT != null && thatT != null)
            {
                this.Visit(thizT, thatT);
                return true;
            }
            else
            {
                Arent();
                return true;
            }
        }

        /// <returns>True if one of the two object was able to be cast to the given type</returns>
        public bool VisitFloatArray(IExpr thiz, IExpr that)
        {
            var thizT = thiz as Tensor<float>;
            var thatT = that as Tensor<float>;
            if (thizT == null && thatT == null)
                return false;
            else if (thizT != null && thatT != null)
            {
                this.Visit(thizT, thatT);
                return true;
            }
            else
            {
                Arent();
                return true;
            }
        }

        public void VisitFloatArrayList<T>(IEnumerable<T> thiz, IEnumerable<T> that)
            where T : Tensor<float> => VisitFloatArrayList(thiz.ToList(), that.ToList());

        public void VisitFloatArrayList<T>(IList<T> thiz, IList<T> that)
            where T: Tensor<float>
        {
            if (!Eq(thiz.Count, that.Count)) return;
            for(int i = 0; i < that.Count; ++i)
            {
                this.Visit(thiz[i], that[i]);
                if (!AreEqual) return;
            }
        }


        public void VisitConst(IConst @const, Scalar<float> data)
        {
            var other = data as Scalar<float>.Const;
            if (other == null)
                Arent();
            else
                Eq(((Scalar<float>.Const)@const).Value, other.Value);
        }

        public void VisitFor(IFor @for, Tensor<float> data)
        {
            var f = (Tensor<float>.For)@for;
            var other = data as Tensor<float>.For;
            if (other == null)
                Arent();
            else
            {
                var loop1 = f.Loop;
                var loop2 = other.Loop;
                And(
                    () => VisitFloatArray(f.Expression, other.Expression),
                    () => VisitFloatArray(f.OutputInfo, other.OutputInfo),
                    () => VisitFloatArrayList(loop1.Sequences.Cast<Tensor<float>>().ToList(), loop2.Sequences.Cast<Tensor<float>>().ToList()),
                    () => VisitFloatArrayList(loop1.Variables.Cast<Tensor<float>>().ToList(), loop2.Variables.Cast<Tensor<float>>().ToList())
                );
            }
        }

        private bool Eq<T>(T thiz, T that)
        {
            var eq = thiz.Equals(that);
            areEqual &= eq;
            return eq;
        }

        //private void And(params Func<bool>[] conditions)
        //{
        //    foreach(var condition in conditions)
        //    {
        //        AreEqual = condition();
        //        if (!AreEqual) return;
        //    }
        //}

        private void And(params Action[] conditions)
        {
            foreach (var condition in conditions)
            {
                condition();
                if (!AreEqual) return;
            }
        }

        public void VisitShared(IShared shared, Scalar<float> data)
        {
            var other = data as Scalar<float>.Shared;
            if (other == null)
                Arent();
            else
                Eq(((Scalar<float>.Shared)shared).Value, other.Value);
        }

        public void VisitShared(IShared shared, Tensor<float> data)
        {
            var other = data as Tensor<float>.Shared;
            if (other == null)
                Arent();
            else
                Eq(((Tensor<float>.Shared)shared).Value, other.Value);
        }

        public void VisitVar(IVar var, Scalar<float> data)
        {
            var other = data as Scalar<float>.Var;
            if (other == null)
                Arent();
            else
                Eq(var.Name, other.Name);
        }

        public void VisitVar(IVar var, Tensor<float> data)
        {
            var other = data as Tensor<float>.Var;
            if (other == null)
                Arent();
            else
                Eq(var.Name, other.Name);
        }

        public void VisitFor(IFor @for, Scalar<float> data)
        {
            throw new NotImplementedException();
        }

        public void VisitConst(IConst @const, Tensor<float> data)
        {
            throw new NotImplementedException();
        }

        public void VisitNAry(IExpr node, Tensor<float> data)
        {
            var other = data as Tensor<float>.NAry;
            if (other == null || node.GetType() != data.GetType())
                Arent();
            else
            {
                // at most one will make recursive calls
                foreach (var (n, o) in (node.Inputs, other.Inputs).Zip())
                {
                    VisitFloat(n, o);
                    VisitFloatArray(n, o);
                }
            }
        }

        public void VisitNAry(IExpr node, Scalar<float> data)
        {
            var other = data as Scalar<float>.NAry;
            if (other == null || node.GetType() != data.GetType())
                Arent();
            else
            {
                // at most one will make recursive calls
                foreach(var (n, o) in (node.Inputs, other.Inputs).Zip())
                {
                    VisitFloat(n, o);
                    VisitFloatArray(n, o);
                }
            }
        }

        public void VisitArray(IArray node, Tensor<float> data)
        {
            throw new NotImplementedException();
        }

        public void VisitArray(IArray node, Scalar<float> data)
        {
            throw new NotImplementedException();
        }

        public void VisitSlice(XSlice node, Tensor<float> data)
        {
            throw new NotImplementedException();
        }

        public void VisitSlice(XSlice node, Scalar<float> data)
        {
            throw new NotImplementedException();
        }
    }

    public static class CanCompareExpressions
    {
        public static bool StructuralEquality(this Tensor<float> thiz, Tensor<float> that)
        {
            var eq = new EqualityComparer();
            eq.Visit(that, thiz);
            return eq.AreEqual && (thiz.ToString() == that.ToString());
        }

        public static bool StructuralEquality(this Scalar<float> thiz, Scalar<float> that)
        {
            var eq = new EqualityComparer();
            eq.Visit(that, thiz);
            return eq.AreEqual && (thiz.ToString() == that.ToString());
        }

        public static void AssertEqual(this Scalar<float> thiz, Scalar<float> that)
        {
            var eq = new EqualityComparer();
            eq.Visit(that, thiz);
            var e1 = thiz.ToString();
            var e2 = that.ToString();
            if (!eq.AreEqual || e1 != e2)
                throw new Exception($"AssertEqual failed. Expected: <{e1}>. Actual: <{e2}>.");
        }

        public static void AssertEqual(this Tensor<float> thiz, Tensor<float> that)
        {
            var eq = new EqualityComparer();
            eq.Visit(that, thiz);
            var e1 = thiz.ToString();
            var e2 = that.ToString();
            if (!eq.AreEqual || e1 != e2)
                throw new Exception($"AssertEqual failed. Expected: <{e1}>. Actual: <{e2}>.");
        }
    }
}
