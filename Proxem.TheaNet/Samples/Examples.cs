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
    // http://deeplearning.net/software/theano/tutorial/examples.html
    public class Examples
    {
        public static void Test1()
        {
            var x = T.Matrix<float>("x");
            var s = 1 / (1 + T.Exp(-x));
            var logistic = T.Function(x, s);
            Console.WriteLine(logistic(NN.Array<float>(new float[,] {
                { 0, 1 },
                { -1, -2 }
            })));

            var s2 = (1 + T.Tanh(x / 2)) / 2;
            var logistic2 = T.Function(x, s2);
            Console.WriteLine(logistic(NN.Array<float>(new float[,] {
                { 0, 1 },
                { -1, -2 }
            })));
        }

        public static void Test2()
        {
            var a = T.Matrix<float>("a");
            var b = T.Matrix<float>("b");
            var diff = a - b;
            var abs_diff = T.Abs(diff);
            var diff_squared = diff * diff;
            var f = T.Function(a, b, new[] { diff, abs_diff, diff_squared });

            var result = f(
                NN.Array<float>(new float[,] { { 1, 1 }, { 1, 1 } }),
                NN.Array<float>(new float[,] { { 0, 1 }, { 2, 3 } })
            );

            Console.WriteLine(result[0]);
            Console.WriteLine(result[1]);
            Console.WriteLine(result[2]);
        }
    }
}
