using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Proxem.TheaNet
{
    public static class EnumerableExtensions
    {
        public static IEnumerable<(T, U)> Zip<T, U>(this (IEnumerable<T> s1, IEnumerable<U> s2) s, Func<T, U, (T, U)> resultSelector)
            => s.s1.Zip(s.s2, resultSelector);

        public static IEnumerable<(T, U)> Zip<T, U>(this (IEnumerable<T> s1, IEnumerable<U> s2) s)
            => s.Zip((x1, x2) => (x1, x2));

        public static IEnumerable<(T, U, V)> Zip<T, U, V>(this (IEnumerable<T> s1, IEnumerable<U> s2, IEnumerable<V> s3) s, Func<T, U, V, (T, U, V)> resultSelector)
            => (s.s1, s.s2).Zip().Zip(s.s3, (x1, x2) => resultSelector(x1.Item1, x1.Item2, x2));

        public static IEnumerable<(T, U, V)> Zip<T, U, V>(this (IEnumerable<T> s1, IEnumerable<U> s2, IEnumerable<V> s3) s)
            => s.Zip((x1, x2, x3) => (x1, x2, x3));
    }
}
