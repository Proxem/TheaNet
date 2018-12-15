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

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Samples
{
    public class Adding
    {
        // http://deeplearning.net/software/theano/tutorial/adding.html

        public static void Test1()
        {
            var x = T.Scalar<float>("x");
            var y = T.Scalar<float>("y");

            var z = x + y;

            var f = T.Function((x, y), z);

            Console.WriteLine(f(2, 3));
            Console.WriteLine(f(16.3f, 12.1f));
        }

        public static void Test2()
        {
            var a = T.Matrix<float>("a");                         // declare variable
            var @out = a + T.Pow(a, 10);                   // build symbolic expression
            var f = T.Function(a, @out);                   // compile function
            Console.WriteLine(f(NN.Array<float>(0, 1, 2)));   // prints `array([0, 2, 1026])`
        }
    }
}
