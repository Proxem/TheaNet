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

using Proxem.TheaNet.Operators.Tensors;

using Dim = Proxem.TheaNet.Scalar<int>;

namespace Proxem.TheaNet
{
    public interface IElementwise { }

    public partial class Tensor<Type>
    {
        /// <summary>Allows to apply a lambda over elements of an array.</summary>
        public class Elementwise : NAry, IElementwise
        {
            // TODO: do we need to hide the `IExpr.Inputs`
            public new Tensor<Type>[] Inputs;

            /// <summary>Variables used in the lambda.</summary>
            public Scalar<Type>.Var[] Vars;

            /// <summary>Expression describing the lambda.</summary>
            public Scalar<Type> Abstraction;

            readonly Dictionary<Tensor<Type>, List<int>> broadcast;

            // this is used to name the lambda variables
            private static int x0 = 0, y0 = 0;

            internal static Tensor<Type> Create(Tensor<Type> x, Func<Scalar<Type>, Scalar<Type>> f)
            {
                if (x is Fill<Type>)
                {
                    var fill = (Fill<Type>)x;
                    return Op.ConstLike(f(fill.x), fill);
                }
                else
                {
                    var _x = new Scalar<Type>.Var($"_x{x0++}");
                    return Create(new[] { x }, new[] { _x }, f(_x));
                }
            }

            public static Tensor<Type> CreateBinary(Tensor<Type> x, Tensor<Type> y, Func<Scalar<Type>, Scalar<Type>, Scalar<Type>> f)
            {
                if (x is Fill<Type>)
                {
                    var fill = (Fill<Type>)x;
                    return Create(y, y_ => f(fill.x, y_));
                }
                else if (y is Fill<Type>)
                {
                    var fill = (Fill<Type>)y;
                    return Create(x, x_ => f(x_, fill.x));
                }

                var _x = new Scalar<Type>.Var($"_x{x0++}");
                var _y = new Scalar<Type>.Var($"_y{y0++}");
                var abstraction = f(_x, _y);

                return Create(new[] { x, y }, new[] { _x, _y }, abstraction);
            }


            public static Tensor<Type> CreateTernary(Tensor<Type> x, Tensor<Type> y, Tensor<Type> z, Func<Scalar<Type>, Scalar<Type>, Scalar<Type>, Scalar<Type>> f)
            {
                var _x = new Scalar<Type>.Var($"_x{x0++}");
                var _y = new Scalar<Type>.Var($"_y{y0++}");
                var _z = new Scalar<Type>.Var($"_z{y0++}");

                var abstraction = f(_x, _y, _z);

                return Create(new[] { x, y, z }, new[] { _x, _y, _z }, abstraction);
            }


            public static Tensor<Type> Create(Tensor<Type>[] inputs, Scalar<Type>.Var[] vars, Scalar<Type> abstraction)
            {
                if (inputs.Length != vars.Length) throw new ArgumentException("Need one captured by inputs");
                var shape = GetShape(inputs);
                // guw: the following code aims at simplifying abstraction like (_x, _y) => _x
                // for now I haven't see a case where this happens, but it might in the future
#if NOT_USED
                var varsInAbstraction = abstraction.FindAll<Scalar<Type>.Var>();
                // easy case, everybody is to be removed
                if (varsInAbstraction.Count == 0)
                {
                    return Op.Const(abstraction, shape);
                }

                var varsToRemove = vars.Where(v => !varsInAbstraction.Contains(v)).ToList();

                if(varsToRemove.Count > 0)
                {
                    inputs = inputs.Where((_, i) => !varsToRemove.Contains(vars[i])).ToArray();
                    vars = vars.Where(v => !varsToRemove.Contains(v)).ToArray();
                }
                if (inputs.Length == 0)
                    return Op.Const(abstraction, shape);
#endif

                // As `Deindexing` operations are heavy we try to apply the lambda before deindexing.
                var deindexing = inputs.All(x => x is Deindexing<Type>);
                if (deindexing)
                {
                    var indices = (inputs.First() as Deindexing<Type>).Indices;
                    var deindexedShape = inputs.First().Shape;
                    if (inputs.All(x => (x as Deindexing<Type>).Indices == indices && x.Shape.WillEqualTo(deindexedShape)))
                    {
                        // TODO not covered
                        var nary = Create(
                            inputs.Select(x => (x as Deindexing<Type>).Content).ToArray(),
                            vars,
                            abstraction
                        );
                        return Deindexing<Type>.Create(nary, deindexedShape, indices);
                    }
                }

                return new Elementwise(inputs, vars, abstraction, shape);
            }

            private static Dim[] GetShape(params Tensor<Type>[] tensors)
            {
                // every one must have the same dim
                var nDim = tensors.First().NDim;
                var shape = new Dim[nDim];

                for (int d = 0; d < nDim; ++d)
                {
                    // first we try to find a fixed size for axis d
                    Dim axis = tensors.Select(x => x.Shape[d] as Dim.Const).FirstOrDefault(x => x?.Value != 1);
                    // if non found we try to find a axis which is unknown
                    axis = axis ?? tensors.Select(x => x.Shape[d]).FirstOrDefault(a => !(a is Dim.Const));
                    // if non found then everyone must be a 1 axis
                    axis = axis ?? 1;
                    shape[d] = axis;
                }
                return shape;
            }

            private Elementwise(Tensor<Type>[] inputs, Scalar<Type>.Var[] vars, Scalar<Type> abstraction, Dim[] shape) :
                base("Elementwise", inputs)
            {
                if (inputs.Length == 0) throw new ArgumentException("Need at least one input");
                if (inputs.Length != vars.Length) throw new ArgumentException("Need one captured by inputs");
                var nDim = inputs.First().NDim;
                if (!inputs.All(x => x.NDim == nDim)) throw new RankException($"Dims don't match: [{string.Join(", ", inputs.Select(_ => _.NDim))}]");

                this.Vars = vars;
                this.Inputs = inputs;

                this.Abstraction = abstraction;
                broadcast = new Dictionary<Tensor<Type>, List<int>>(inputs.Length);
                foreach (var x in inputs) broadcast[x] = new List<int>();

                // checks and binds shape
                Shape = shape;
                for (int d = 0; d < nDim; ++d)
                    foreach (var x in inputs)
                        if (x.Shape[d].NeedBroadcast(this.Shape[d]))
                            broadcast[x].Add(d);
                        else
                            ShapeExtension.Bind(ref this.Shape[d], ref x.Shape[d]);
            }

            public override sealed Dim[] Shape { get; }

            public override sealed void Backward(Tensor<Type> delta, Backpropagation bp)
            {
                delta.AssertOfShape(Shape);

                for (int i = 0; i < Inputs.Length; ++i)
                {
                    var x = Inputs[i];
                    var deltaX = delta * D(i);
                    foreach (int axis in broadcast[x])
                        deltaX = Op.Sum(deltaX, axis, keepDims: true);
                    bp.PushGradientTo(x, deltaX);
                }

                // TODO: fix. This may push gradient using variables of the lambda outside of the Apply.
                bp.PushGradientTo(Abstraction, Op.Sum(delta));
            }

            private Backpropagation bp = null;

            protected Tensor<Type> D(int i)
            {
                bp = bp ?? Backpropagation.Backward(Abstraction, Numeric<Type>.One);
                var _x = Vars[i];
                var d_x = (Scalar<Type>)bp.ScalarDerivatives[_x];
                return Lift(d_x);
            }

            /// TODO: if `Lambda` was a true Expr, this could probably be remove
            public override void Process(IProcessor processor) => processor.ProcessElementwise(this);

            /// <summary>
            /// Creates a Tensor from an Abstraction.
            /// The difference with Elementwise.Create is that we reuse the existing arrays.
            /// </summary>
            private Tensor<Type> Lift(Scalar<Type> expr)
            {
                if (expr == this.Abstraction) return this;

                var result = Vars.FirstOrDefault(e => e == expr);
                if (result != null) return Inputs[Array.IndexOf(Vars, expr)];

                // HACK: here the `Clone` only works on some specific cases.
                switch (expr)
                {
                    case Scalar<Type>.Binary binary:
                        return Op.Apply(Lift(binary.x), Lift(binary.y), (_x, _y) => binary.Clone(_x, _y));
                    case Scalar<Type>.Unary unary:
                        return Op.Apply(Lift(unary.x), _x => unary.Clone(_x));
                    case Scalar<Type>.Const cst:
                        return Op.ConstLike(expr, this);
                    default:
                        throw new NotImplementedException();
                }
            }

            public override NAry Clone(IReadOnlyList<IExpr> inputs)
            {
                var newVars = Vars.Select(v => new Scalar<Type>.Var(v.Name)).ToArray();
                var patch = new Patch();
                for (int i = 0; i < Vars.Length; ++i)
                    patch[Vars[i]] = newVars[i];

                var abstraction = (Scalar<Type>)Abstraction.Patch(patch);
                return (NAry)Create(inputs.Cast<Tensor<Type>>().ToArray(), newVars, abstraction);
            }
        }
    }
}
