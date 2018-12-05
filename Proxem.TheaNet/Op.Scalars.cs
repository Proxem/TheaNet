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
using Proxem.TheaNet.Operators.Scalars;
using Proxem.TheaNet.Operators.FloatScalars;

namespace Proxem.TheaNet
{
    public partial class Op
    {
        private static void CheckName(string name)
        {
            if (name.StartsWith("_")) throw new Exception("Names should not start with underscore");
        }

        public static Scalar<T>.Var Scalar<T>(string name)
        {
            CheckName(name);
            return new Scalar<T>.Var(name);
        }

        public static Scalar<T>.Shared Shared<T>(T v, string name)
        {
            CheckName(name);
            return new Scalar<T>.Shared(v, name);
        }

        public static Scalar<T>.Const Const<T>(T x) => TheaNet.Scalar<T>.Const.Create(x);

        public static Scalar<float>.Unary Exp(Scalar<float> x) => new Scalar<float>.Unary("Exp", x, dx: (_x, _f) => _f);

        public static Scalar<T> ReLu<T>(Scalar<T> x) => new Scalar<T>.Unary("ReLu", x, dx: (_x, _f) => (_f > Numeric<T>.Zero));

        public static Scalar<float>.Unary Log(Scalar<float> x) => new Scalar<float>.Unary("Log", x, dx: (_x, _f) => 1 / _x);

        public static Scalar<float>.Unary Clip(Scalar<float> x, float min, float max) =>
            new Scalar<float>.Unary("Clip", x, dx: (_x, _f) => 1, extraInputs: new object[] { min, max });

        public static Scalar<float>.Unary Sigmoid(Scalar<float> x) => new Scalar<float>.Unary("Sigmoid", x, dx: (_x, _f) => (1 - _f) * _f);

        // TODO: fix derivative
        public static Scalar<float> Max(Tensor<float> x) => new Aggregate<float>("Max", x, dx: (_x, _f) => Op.ZerosLike(x));

        public static Scalar<T> Max<T>(Scalar<T> x, Scalar<T> y) =>
            new Scalar<T>.Binary("Max", x, y,
                dx: (_x, _y, _f) => x > y,
                dy: (_x, _y, _f) => y > x
            );

        public static Scalar<T> Min<T>(Scalar<T> x, Scalar<T> y) =>
            new Scalar<T>.Binary("Min", x, y,
                dx: (_x, _y, _f) => x < y,
                dy: (_x, _y, _f) => y < x
            );

        public static Scalar<int> Mod(Scalar<int> x, Scalar<int> y) =>
           new Scalar<int>.Binary("Mod", x, y, dx: null, dy: null);

        public static Scalar<int> Argmax<T>(Tensor<T> x) => new Argmax<T>(x);

        public static Scalar<float> Mean(Tensor<float> x) => Sum(x) / x.Size.As<float>();

        public static Scalar<float> Mean(Tensor<int> x) => Sum(x).As<float>() / x.Size.As<float>();

        public static Scalar<float> Norm2(Tensor<float> x) => new Aggregate<float>("Norm2", x, dx: (_x, _f) => 2 * _x);

        // TODO: rename as ScalarDot
        public static Scalar<float> VectorDot(Tensor<float> x, Tensor<float> y)
        {
            return (Scalar<float>)Operators.FloatTensors.Dot.Create(x, y);
        }

        public static Scalar<float> Pow(Scalar<float> x, float a)
        {
            if (a == 1f) return x;
            else if (a == 0f) return 1f;
            else if (a == 2f) return x * x;
            else return new Pow(x, a);
        }

        public static Scalar<float> Pow(Scalar<float> x, Scalar<float> y)
        {
            if (y is Scalar<float>.Const consy) return Pow(x, consy.Value);
            return new Pow(x, y);
        }

        public static Scalar<float> Pow(float a, Scalar<float> y)
        {
            if (y is Scalar<float>.Const consy) return (float)Math.Pow(a, consy.Value);
            return new Pow(a, y);
        }

        /// <summary> Identity with the side effect of printing the given value to the console. </summary>
        public static Scalar<T> Print<T>(Scalar<T> x) => Print(null, x);

        public static Scalar<T> Print<T>(string format, Scalar<T> x) => new Scalar<T>.Unary("Print", x, (_, f) => Numeric<T>.One, new[] { format });

        public static Scalar<T> Square<T>(Scalar<T> x)
        {
            if (x.IsZero) return x;
            if (x.IsOne) return x;
            if (x.IsMinusOne) return Numeric<T>.One;
            return new Scalar<T>.Unary("Square", x, dx: (_x, _f) => Numeric<T>.Two * x);
        }

        public static Scalar<float>.Unary Sqrt(Scalar<float> x) => new Scalar<float>.Unary("Sqrt", x, dx: (_x, _f) => 1 / (2 * _f));

        public static Scalar<Type> Sum<Type>(Tensor<Type> x)
        {
            switch (x)
            {
                case Operators.Tensors.OneHotPoint<Type> hot:
                    return hot.Content;
                default:
                    return new Aggregate<Type>("Sum", x, dx: (_x, _f) => Op.OnesLike(_x));
            }
        }

        public static Scalar<float>.Unary Tanh(Scalar<float> x) => new Scalar<float>.Unary("Tanh", x, dx: (_x, _f) => 1 - _f * _f);

        public static Scalar<Type> Abs<Type>(Scalar<Type> x)
        {
            switch (x)
            {
                case Scalar<Type>.Const @const:
                    return (Scalar<Type>)Numeric.Abs(@const.Value);
                //case Neg<Type> neg:
                //    return Abs(neg.x);
                default:
                    return new Scalar<Type>.Unary("Abs", x, dx: (_x, _f) => (x > Numeric<Type>.Zero) - (x < Numeric<Type>.Zero));
            }
        }

        public static Scalar<T> Sign<T>(Scalar<T> x) => new Scalar<T>.Unary("Sign", x, dx: (_x, _f) => Numeric<T>.Zero);

        public static Scalar<T> Eq<T>(Scalar<T> x, Scalar<T> y) => (x >= y) * (x <= y);

        public static Scalar<float> LogisticLoss(Scalar<float> gold, Scalar<float> pred) => -gold * Log(pred) - (1 - gold) * Log(1 - pred);
        public static Scalar<float> LogisticLoss(Scalar<int> gold, Scalar<float> pred) => (-gold).As<float>() * Log(pred) - (1 - gold).As<float>() * Log(1 - pred);

        public static Scalar<float> DirichletPdf(Tensor<float> x, float alpha) =>
            new DirichletPdf(alpha, x);

        public static Scalar<R> Switch<T, R>(Scalar<T> mask, Scalar<R> ifTrue, Scalar<R> ifFalse) => Operators.Scalars.Switch.Create(mask, ifTrue, ifFalse);
    }
}
