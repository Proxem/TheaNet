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
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Proxem.NumNet;
using Proxem.TheaNet.Operators.Tensors;


namespace Proxem.TheaNet.Test
{
    [TestClass]
    public class TestTuple
    {
        class TupleSS : Tuple2, ITuple<Scalar<float>, float, Scalar<float>, float>, IExpr<(float, float)>
        {
            public TupleSS(Scalar<float> x, Scalar<float> y) : base(x, y) { }

            public Scalar<float> x => (Scalar<float>)Inputs[0];
            public Scalar<float> y => (Scalar<float>)Inputs[1];

            public override void Process(IProcessor processor) => processor.ProcessFunctionCall<(float, float)>(this, "Tuple.Create");

            public void Backward1(Scalar<float> delta, Backpropagation bp) => bp.PushGradientTo(x, delta);

            public void Backward2(Scalar<float> delta, Backpropagation bp) => bp.PushGradientTo(y, delta);

            public override Tuple2 Clone(IReadOnlyList<IExpr> inputs) => new TupleSS((Scalar<float>)inputs[0], (Scalar<float>)inputs[1]);
        }

        [TestMethod]
        public void CanForwardScalarScalarTuple()
        {
            var x = Op.Scalar<float>("x");
            var x_succx = new TupleSS(x, x + 1);
            var x_ = x_succx.Item1();
            var succx = x_succx.Item2();
            var xpxp1 = x_ + succx;

            var f = Op.Function(input: x, output: xpxp1);
            Assert.AreEqual(1, f(0));
            Assert.AreEqual(3, f(1));
            Assert.AreEqual(13, f(6));
        }

        [TestMethod]
        public void CanBackwardScalarScalarTuple()
        {
            var x = Op.Scalar<float>("x");
            var x_succx = new TupleSS(x, x + 1);
            var x_ = x_succx.Item1();
            var succx = x_succx.Item2();
            var xpxp1 = x_ + succx;

            var df = Op.Function(input: x, output: Op.Grad(xpxp1, x));
            Assert.AreEqual(2, df(0));
            Assert.AreEqual(2, df(1));
            Assert.AreEqual(2, df(6));
        }

        public class TupleTT :
            Tuple2, ITuple<Tensor<float>, Array<float>, Tensor<float>, Array<float>>,
            ITensorTuple
        {
            public TupleTT(Tensor<float> x, Tensor<float> y) : base(x, y) { }

            public override void  Process(IProcessor processor) => processor.ProcessFunctionCall<(Array<float>, Array<float>)>(this, "Tuple.Create");

            public Tensor<float> x => (Tensor<float>)Inputs[0];
            public Tensor<float> y => (Tensor<float>)Inputs[1];

            public void Backward1(Tensor<float> delta, Backpropagation bp) => bp.PushGradientTo(x, delta);

            public void Backward2(Tensor<float> delta, Backpropagation bp) => bp.PushGradientTo(y, delta);

            Scalar<int>[] ITensorTuple.Shape(int item)
            {
                if (item == 1) return x.Shape;
                else if (item == 2) return y.Shape;
                else throw new IndexOutOfRangeException();
            }

            public override Tuple2 Clone(IReadOnlyList<IExpr> inputs) => new TupleTT((Tensor<float>)inputs[0], (Tensor<float>)inputs[1]);
        }

        [TestMethod]
        public void CanForwardTensorTensorTuple()
        {
            var x = Op.Vector<float>("x");
            var x_x2 = new TupleTT(x, x * x);
            var x_ = x_x2.Item1();
            var x2 = x_x2.Item2();
            var xpx2p1 = x_ + x2 + 1;
            var loss = Op.Sum(xpx2p1);

            var f = Op.Function(input: x, output: loss);
            Assert.AreEqual(11, f(NN.Range<float>(3)));
            Assert.AreEqual(4, f(NN.Zeros<float>(4)));
        }

        [TestMethod]
        public void CanBackwardTensorTensorTuple()
        {
            var x = Op.Vector<float>("x");
            var x_x2 = new TupleTT(x, x * x);
            var x_ = x_x2.Item1();
            var x2 = x_x2.Item2();
            var xpx2p1 = x_ + x2 + 1;
            var loss = Op.Sum(xpx2p1);

            var df = Op.Function(input: x, output: Op.Grad(loss, x));
            var y = NN.Zeros(4);
            AssertArray.AreEqual(2 * y + 1, df(y));
            y = NN.Range<float>(3);
            AssertArray.AreEqual(2 * y + 1, df(y));
        }

        public class TupleTS :
            Tuple2, ITuple<Tensor<float>, Array<float>, Scalar<float>, float>,
            ITensorTuple
        {
            public TupleTS(Tensor<float> x, Scalar<float> y) : base(x, y) {}

            public Tensor<float> x => (Tensor<float>)Inputs[0];
            public Scalar<float> y => (Scalar<float>)Inputs[1];

            public Scalar<int>[] Shape(int item)
            {
                if (item != 1) throw new IndexOutOfRangeException();
                else return x.Shape;
            }

            public void Backward1(Tensor<float> delta, Backpropagation bp) => bp.PushGradientTo(x, delta);

            public void Backward2(Scalar<float> delta, Backpropagation bp) => bp.PushGradientTo(y, delta);

            public override void Process(IProcessor processor) => processor.ProcessFunctionCall<(Array<float>, float)>(this, "Tuple.Create");

            public override Tuple2 Clone(IReadOnlyList<IExpr> inputs) => new TupleTS((Tensor<float>)inputs[0], (Scalar<float>)inputs[1]);
        }

        [TestMethod]
        public void CanForwardTensorScalarTuple()
        {
            var x = Op.Vector<float>("x");
            var x2_sum = new TupleTS(x * x, Op.Sum(x));
            var x2 = x2_sum.Item1();
            var sum = x2_sum.Item2();
            var loss = Op.Sum(x2) + sum;

            var f = Op.Function(input: x, output: loss);
            Assert.AreEqual(8, f(NN.Array<float>(1, 2, -1)));
            Assert.AreEqual(8, f(NN.Range<float>(3)));
        }

        [TestMethod]
        public void CanBackwardTensorScalarTuple()
        {
            var x = Op.Vector<float>("x");
            var x2_sum = new TupleTS(x * x, Op.Sum(x));
            var x2 = x2_sum.Item1();
            var sum = x2_sum.Item2();
            var loss = Op.Sum(x2) + sum;

            var df = Op.Function(input: x, output: Op.Grad(loss, x));
            var y = NN.Zeros(4);
            AssertArray.AreEqual(2 * y + 1, df(y));
            y = NN.Range<float>(3);
            AssertArray.AreEqual(2 * y + 1, df(y));
        }
    }
}
