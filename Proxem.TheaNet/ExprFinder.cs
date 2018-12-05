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

namespace Proxem.TheaNet
{
    public enum TraverseMode { RECURSIVE, STOP_BEFORE_VISITED, STOP_ON_VISITED }

    public static class IExprExtension
    {
        public static HashSet<T> FindAll<T>(this IExpr expr) where T : class, IExpr
        {
            var all = new HashSet<T>();
            expr.Traverse(e => { if (e is T) all.Add((T)e); }, mode: TraverseMode.STOP_BEFORE_VISITED);
            return all;
        }

        public static HashSet<T> FindAll<T>(this IExpr expr, Func<T, bool> f) where T : class, IExpr
        {
            var all = new HashSet<T>();
            expr.Traverse(e => { if (e is T && f((T)e)) all.Add((T)e); }, mode: TraverseMode.STOP_BEFORE_VISITED);
            return all;
        }

        public static IExpr Map(this IExpr expr, Func<IExpr, IExpr> f)
        {
            var dic = new Dictionary<IExpr, IExpr>();
            _fillDic(expr, f, dic);
            return expr.Patch(new Patch(dic));
        }

        private static void _fillDic(IExpr expr, Func<IExpr, IExpr> f, Dictionary<IExpr, IExpr> dic)
        {
            if (dic.ContainsKey(expr)) return;
            dic[expr] = f(expr);
            foreach (var e in expr.Inputs)
                _fillDic(e, f, dic);
        }

        public static void Traverse(this IExpr expr, Action<IExpr> f, bool postfix = false, TraverseMode mode = TraverseMode.RECURSIVE)
        {
            bool distinct = mode != TraverseMode.RECURSIVE;
            if (distinct)
                _traverseDistinct(expr, f, new HashSet<IExpr>(), postfix, mode);
            else
                _traverse(expr, f, postfix);
        }

        private static void _traverseDistinct(IExpr expr, Action<IExpr> f, HashSet<IExpr> dic, bool postfix, TraverseMode mode)
        {
            bool visited = dic.Contains(expr);
            if (mode == TraverseMode.STOP_BEFORE_VISITED && visited)
                return;

            if (!postfix)
            {
                f(expr);
                dic.Add(expr);
                if (mode == TraverseMode.STOP_ON_VISITED && visited)
                    return;
            }

            foreach (var e in expr.Inputs)
                _traverseDistinct(e, f, dic, postfix, mode);

            if (postfix)
            {
                f(expr);
                dic.Add(expr);
            }
        }

        private static void _traverse(IExpr expr, Action<IExpr> f, bool postfix)
        {
            if(!postfix)
                f(expr);
            foreach (var e in expr.Inputs)
                _traverse(e, f, postfix);
            if (postfix)
                f(expr);
        }

        public static void Traverse(this IExpr expr, Func<IExpr, bool> preStop = null, Action<IExpr> preAction = null, Func<IExpr, bool> postStop = null, Action<IExpr> postAction = null)
        {
            if (preStop != null && preStop(expr)) return;

            preAction?.Invoke(expr);

            if (postStop != null && postStop(expr)) return;

            foreach (var e in expr.Inputs)
                e.Traverse(preStop, preAction, postStop, postAction);

            postAction?.Invoke(expr);
        }
    }
}
