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
using Proxem.NumNet;

namespace Proxem.TheaNet
{
    public class Numeric<Type>
    {
        public static readonly Type Zero = FromInt(0);
        public static readonly Type One = FromInt(1);
        public static readonly Type Two = FromInt(2);
        public static readonly Type MinusOne = FromInt(-1);

        public static Type FromInt(int v)
        {
            return (Type)Convert.ChangeType(v, typeof(Type));
        }

        public static Numeric<Type> Current;

        static Numeric()
        {
            var name = "Proxem.TheaNet.Numerics." + typeof(Type).Name;
            var type = System.Reflection.Assembly.GetExecutingAssembly().GetType(name, throwOnError: false);
            if (type != null)
            {
                Current = (Numeric<Type>)type.GetConstructor(new System.Type[0]).Invoke(null);
            }
            else Current = new Numeric<Type>();
        }

        public virtual string GetLiteral(Type a)
        {
            throw new InvalidOperationException();
        }

        public virtual bool IsNegative(Type a)
        {
            throw new InvalidOperationException();
        }

        public virtual Type Neg(Type a)
        {
            throw new InvalidOperationException();
        }

        public virtual Type Add(Type a, Type b)
        {
            throw new InvalidOperationException();
        }

        public virtual Type Sub(Type a, Type b)
        {
            throw new InvalidOperationException();
        }

        public virtual Type Mul(Type a, Type b)
        {
            throw new InvalidOperationException();
        }

        public virtual Type Div(Type a, Type b)
        {
            throw new InvalidOperationException();
        }

        public virtual bool IntegerDiv()
        {
            throw new InvalidOperationException();
        }

        public virtual Type GetScalar(string name)
        {
            throw new InvalidOperationException();
        }

        public virtual void SetScalar(string name, Type value)
        {
            throw new InvalidOperationException();
        }

        public virtual Array<Type> GetTensor(string name)
        {
            throw new InvalidOperationException();
        }

        public virtual void SetTensor(string name, Array<Type> value)
        {
            throw new InvalidOperationException();
        }

        public virtual Type Abs(Type a)
        {
            throw new InvalidOperationException();
        }
    }

    public static class Numeric
    {
        public static string GetLiteral<Type>(Type a)
        {
            return Numeric<Type>.Current.GetLiteral(a);
        }

        public static bool IsNegative<Type>(Type a)
        {
            return Numeric<Type>.Current.IsNegative(a);
        }

        public static Type Neg<Type>(Type a)
        {
            return Numeric<Type>.Current.Neg(a);
        }

        public static Type Add<Type>(Type a, Type b)
        {
            return Numeric<Type>.Current.Add(a, b);
        }

        public static Type Sub<Type>(Type a, Type b)
        {
            return Numeric<Type>.Current.Sub(a, b);
        }

        public static Type Mul<Type>(Type a, Type b)
        {
            return Numeric<Type>.Current.Mul(a, b);
        }

        public static Type Div<Type>(Type a, Type b)
        {
            return Numeric<Type>.Current.Div(a, b);
        }

        public static Type GetScalar<Type>(string name)
        {
            return Numeric<Type>.Current.GetScalar(name);
        }

        public static void SetScalar<Type>(string name, Type value)
        {
            Numeric<Type>.Current.SetScalar(name, value);
        }

        public static Array<Type> GetTensor<Type>(string name)
        {
            return Numeric<Type>.Current.GetTensor(name);
        }

        public static void SetTensor<Type>(string name, Array<Type> value)
        {
            Numeric<Type>.Current.SetTensor(name, value);
        }

        public static Type Abs<Type>(Type a)
        {
            return Numeric<Type>.Current.Abs(a);
        }
    }
}
