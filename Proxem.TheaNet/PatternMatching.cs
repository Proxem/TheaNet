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
using System.Text;
using System.Threading.Tasks;

namespace Proxem.TheaNet
{
    public static class PatternMatching
    {
        //[Obsolete("use switch")]
        public static R Match<T, T1, R>(this T thiz, Func<T1, R> f)
            where T1 : class, T
            where R : class
        {
            return (thiz as T1).Map(f);
        }

        //[Obsolete("use switch")]
        public static R Match<T, T1, R>(this T thiz, Func<T1, R> f1, Func<R> @default)
            where T1 : class, T
        {
            var t1 = thiz as T1;
            if (t1 != null) return f1(t1);
            else if(@default != null) return @default();
            else
                throw new NotImplementedException($"Match not exhaustif. Type {typeof(T)} can also be a {thiz.GetType()}.");
        }

        //[Obsolete("use switch")]
        public static R Match<T, T1, T2, R>(this T thiz, Func<T1, R> f1, Func<T2, R> f2, Func<R> @default)
            where T1 : class, T
            where T2 : class, T
        {
            var t1 = thiz as T1;
            if (t1 != null) return f1(t1);
            else return thiz.Match(f2, @default);
        }

        //[Obsolete("use switch")]
        public static R Match<T, T1, T2, T3, R>(this T thiz, Func<T1, R> f1, Func<T2, R> f2, Func<T3, R> f3, Func<R> @default)
            where T1 : class, T
            where T2 : class, T
            where T3 : class, T
        {
            var t1 = thiz as T1;
            if (t1 != null) return f1(t1);
            else return thiz.Match(f2, f3, @default);
        }

        //[Obsolete("use switch")]
        public static R Match<T, T1, T2, T3, T4, R>(this T thiz, Func<T1, R> f1, Func<T2, R> f2, Func<T3, R> f3, Func<T4, R> f4, Func<R> @default)
            where T : class
            where T1 : class, T
            where T2 : class, T
            where T3 : class, T
            where T4 : class, T
        {
            var t1 = thiz as T1;
            if (t1 != null) return f1(t1);
            else return thiz.Match(f2, f3, f4, @default);
        }

        //[Obsolete("use switch")]
        public static R Match<T, T1, T2, T3, T4, T5, R>(this T thiz, Func<T1, R> f1, Func<T2, R> f2, Func<T3, R> f3, Func<T4, R> f4, Func<T5, R> f5, Func<R> @default)
            where T : class
            where T1 : class, T
            where T2 : class, T
            where T3 : class, T
            where T4 : class, T
            where T5 : class, T
        {
            var t1 = thiz as T1;
            if (t1 != null) return f1(t1);
            else return thiz.Match(f2, f3, f4, f5, @default);
        }

        public static T Enforce<T>(this T thiz, Func<T, bool> condition)
            where T: class
        {
            return (thiz != null && condition(thiz)) ? thiz : null;
        }

        public delegate bool TryFunction<T, R>(T input, out R result);

        /// <summary>
        /// Wrapper around 'out' functions.
        /// </summary>
        /// <typeparam name="T">The input type of the function</typeparam>
        /// <typeparam name="R">The output type of the function</typeparam>
        /// <param name="get">The 'out' function</param>
        /// <param name="input">The input</param>
        /// <param name="else">The value to use if the function returns false</param>
        /// <returns>The 'out' value of the function in case of success, 'else' otherwise</returns>
        public static R GetOrElse<T, R>(TryFunction<T, R> @get, T input, R @else)
        {
            R result;
            if (@get(input, out result))
                return result;
            else
                return @else;
        }

        public static R GetOrElse<T, R>(TryFunction<T, R> @get, T input, Func<R> @else)
        {
            R result;
            if (@get(input, out result))
                return result;
            else
                return @else();
        }

        public static T1 Enforce<T, T1>(this T thiz, Func<T1, bool> condition)
            where T1 : class
        {
            var t1 = thiz as T1;
            return t1 != null && condition(t1) ? t1 : null;
        }

        public static bool Check<T, T1>(this T thiz, Func<T1, bool> condition)
            where T1 : class
        {
            var t1 = thiz as T1;
            return t1 != null && condition(t1);
        }

        //[Obsolete("use switch")]
        public static T1 Match<T, T1>(this T thiz, Func<T1, bool> condition)
            where T : class
            where T1 : class
        {
            var t1 = thiz as T1;
            return (t1 != null && condition(t1)) ? t1 : null;
        }

        //[Obsolete("use switch")]
        public static R Match<T, T1, R>(this T thiz, Func<T1, bool> condition, Func<T1, R> f)
            where T : class
            where T1 : class
            where R : class
        {
            var t1 = thiz as T1;
            return (t1 != null && condition(t1)) ? f(t1) : null;
        }

        public static R TakeFirst<R>(params Func<R>[] fs)
        {
            var y = default(R);
            foreach (var f in fs)
            {
                y = f();
                if (y != null) return y;
            }
            return y;
        }

        public static R Map<T, R>(this T thiz, Func<T, R> f)
            where T: class
            where R: class
        {
            return thiz == null ? null : f(thiz);
        }
    }
}
