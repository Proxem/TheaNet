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
using Proxem.NumNet.Single;

using T = Proxem.TheaNet.Op;

namespace Proxem.TheaNet.Samples
{
    // http://deeplearning.net/software/theano/tutorial/loop.html
    // http://deeplearning.net/software/theano/library/scan.html
    public class Scan
    {
        public static void TestLook1()
        {
            // defining the tensor variables
            var X = T.Matrix<float>("x");
            var W = T.Matrix<float>("W");
            var b_sym = T.Matrix<float>("b_sym");

            var results = T.Scan(v => T.Tanh(T.Dot(v, W) + b_sym), sequence: X);
            var compute_elementwise = T.Function(inputs: new[] { X, W, b_sym }, output: results);

            // test values
            var x = NN.Eye<float>(2);
            var w = NN.Ones<float>(2, 2);
            var b = NN.Ones<float>(2);
            b.Item[1] = 2;

            Console.WriteLine(compute_elementwise(new[] { x, w, b }).Item[0]);

            // comparison with tensors
            Console.WriteLine(NN.Tanh(x.Dot(w) + b));
        }

        public static void TestScan1()
        {
            var k = Op.Scalar<float>("k");
            var A = Op.Matrix<float>("A");

            // Symbolic description of the result
            //var scan = Op.Scan((prior_result, AA) => prior_result * AA,
            //                  outputs_info: new TensorExpr.Fill(1, A),
            //                  non_sequences: A,
            //                  n_steps: k);
            //var result = scan.Item1;
            //var updates = scan.Item2;

            // compiled function that returns A**k
            //var power = Op.Function(A, k,
            //    outputs: result,
            //    updates: updates);

            //Console.WriteLine(power(ArrayOp.Range(10), 2));
            //Console.WriteLine(power(ArrayOp.Range(10), 4));
        }
    }
}
