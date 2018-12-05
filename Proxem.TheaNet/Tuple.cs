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

using Proxem.NumNet;

namespace Proxem.TheaNet
{
    using Dim = Scalar<int>;
    using Tuple_ = Tuple<object>;

    public interface ITuple : IExpr<Tuple_> {
        //void Backward(int item, object delta, Backpropagation bp);
    }

    public interface ITuple<A, A_, B, B_> : IExpr<Tuple<A_, B_>>, ITuple//, ITuple1<A, A_>, ITuple2<B, B_>
        where A : class, IExpr<A_>
        where B : class, IExpr<B_>
    {
        void Backward1(A delta, Backpropagation bp);
        void Backward2(B delta, Backpropagation bp);
    }

    /// <summary>
    /// Every tuple containing at least one Tensor must implement this interface,
    /// and provide the shape of these Tensor, and throw Exception for other item.
    /// </summary>
    public interface ITensorTuple: ITuple
    {
        Dim[] Shape(int item);
    }

    public abstract class Tuple2: Expr, ITuple
    {
        public abstract Tuple2 Clone(IReadOnlyList<IExpr> inputs);

        public Tuple2(params IExpr[] inputs): base("Tuple2", inputs) { }

        public override sealed IExpr Patch(Patch substitutions)
        {
            Tuple2 result;
            if (substitutions.TryGetValue(this, out result)) return result;
            var patchInputs = Inputs.Patch(substitutions);
            if (patchInputs == Inputs) return this;
            result = Clone(patchInputs);
            substitutions.Add(this, result);
            return result;
        }
    }

    public static class ITupleExtension
    {
        public static void Backward(this ITuple thiz, int item, object delta, Backpropagation bp)
        {
            dynamic x = thiz;
            _Backward(x, item, delta, bp);
        }

        private static void _Backward<A, A_, B, B_>(ITuple<A, A_, B, B_> thiz, int item, object delta, Backpropagation bp)
            where A : class, IExpr<A_>
            where B : class, IExpr<B_>
        {
            if (item == 1) thiz.Backward1((A)delta, bp);
            else if (item == 2) thiz.Backward2((B)delta, bp);
            else throw new ArgumentException(string.Format("There is no item {0} in this tuple-2.", item));
        }

        public static ScalarItem<A_> Item1<A_, B, B_>(this ITuple<Scalar<A_>, A_, B, B_> tuple)
            where B : class, IExpr<B_>
        {
            return new ScalarItem<A_>(tuple, 1);
        }

        public static TensorItem<A_> Item1<A_, B, B_>(this ITuple<Tensor<A_>, Array<A_>, B, B_> tuple)
            where B : class, IExpr<B_>
        {
            return new TensorItem<A_>((ITensorTuple)tuple, 1);
        }

        //public static TensorItem<A_> Item1<A_, B, B_, T>(this T tuple)
        //    where T: ITuple<Tensor<A_>, Array<A_>, B, B_>, ITupleTensor
        //    where B : class, IExpr<B_>
        //{
        //    return new TensorItem<A_>(tuple, 1);
        //}

        public static ScalarItem<B_> Item2<A, A_, B_>(this ITuple<A, A_, Scalar<B_>, B_> tuple)
            where A : class, IExpr<A_>
        {
            return new ScalarItem<B_>(tuple, 2);
        }

        public static TensorItem<B_> Item2<A, A_, B_>(this ITuple<A, A_, Tensor<B_>, Array<B_>> tuple)
            where A : class, IExpr<A_>
        {
            return new TensorItem<B_>((ITensorTuple) tuple, 2);
        }

        //public static TensorItem<B_> Item2<A, A_, B_, T>(this T tuple)
        //    where T: ITuple<A, A_, Tensor<B_>, Array<B_>>, ITupleTensor
        //    where A : class, IExpr<A_>
        //{
        //    return new TensorItem<B_>(tuple, 2);
        //}
    }

    public interface TupleElement<A, A_> : IExpr<A_>
        where A : class, IExpr<A_>
    {
        int ItemIndex { get; }
    }

    public class ScalarItem<A_> : Scalar<A_>.NAry, TupleElement<Scalar<A_>, A_>
    {
        internal ScalarItem(ITuple parent, int itemIndex): base("TupleItem", new[] { parent }, new object[] { itemIndex })
        {
        }

        public ITuple Parent => (ITuple)this.Inputs.First();

        public int ItemIndex => (int)this._extraInputs[0];

        public override void Backward(Scalar<A_> delta, Backpropagation bp) =>
            ITupleExtension.Backward(Parent, ItemIndex, delta, bp);

        public override Scalar<A_> Clone(IReadOnlyList<IExpr> inputs) => new ScalarItem<A_>((ITuple)inputs[0], ItemIndex);
    }

    public class TensorItem<A_> : Tensor<A_>.Unary<ITensorTuple, Tuple_>, TupleElement<Tensor<A_>, Array<A_>>
    {
        readonly Dim[] _shape;

        internal TensorItem(ITensorTuple parent, int itemIndex): base("TupleItem", parent, itemIndex)
        {
            this.ItemIndex = itemIndex;
            _shape = x.Shape(itemIndex);
        }

        public int ItemIndex { get; }

        public override void Backward(Tensor<A_> delta, Backpropagation bp) =>
            ITupleExtension.Backward(x, ItemIndex, delta, bp);

        public override Unary<ITensorTuple, Tuple_> Clone(ITensorTuple x) =>
            new TensorItem<A_>(x, ItemIndex);

        public override Dim[] Shape => _shape;

        //ITuple IUnary<ITuple, Tuple_>.x => x;
    }
}
