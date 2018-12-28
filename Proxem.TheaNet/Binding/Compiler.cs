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

using Microsoft.CSharp;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;

namespace Proxem.TheaNet.Binding
{
    /// <summary>
    /// Generates a compilable C# class hosting the code for a function, from TheaNet expressions.
    /// </summary>
    public class Compiler
    {
#if DEBUG
        public static bool Debug = true;
#else
        public static bool Debug = false;
#endif

#if VERBOSE
        public bool Verbose = true;
#else
        public bool Verbose = false;
#endif

#if MEM
        public bool Mem = true;
#else
        public bool Mem = false;
#endif

        public StringBuilder sb;
        public int ID;
        public Scope Scope;
        public int Tab;
        public int CurrentLine = 0;

        private MemoryAllocator mem = null;
        private bool lockDecount = false;

        public IDictionary<string, Delegate> CustomFunctions = new Dictionary<string, Delegate>();

        public Compiler()
        {
            this.sb = new StringBuilder();
        }

        public string GetSource() => sb.ToString();

        /// <summary>
        /// Extract the arguments of the function.
        /// If `listName` is not null the arguments must be extracted from a list.
        /// If `arrayName` is not null the arguments must be extracted from an array.
        /// The difference is due to list.Count and array.Length
        /// </summary>
        public void CompileArgs(IList<Tuple<IVar, Type>> args, string listName, string arrayName)
        {
            if (listName == null && arrayName == null)
            {
                foreach (var arg in args)
                {
                    //CompileArg(arg.Item1);
                    var var = arg.Item1;
                    Scope.Declare(var, this, var.Name);
                    CheckDim(var);
                }
            }
            else
            {
                var fromList = listName != null;
                if (fromList && arrayName != null) throw new ArgumentException();

                var name = listName ?? arrayName;
                var length = fromList ? "Count" : "Length";
                EmitLine($"Assert({name}.{length} == {args.Count});");

                int i = 0;
                foreach (var arg in args)
                {
                    //CompileArg(arg.Item1, $"({arg.Item2.GetName()}){name}[{i++}]");
                    var var = arg.Item1;
                    var expr = $"({arg.Item2.GetName()}){name}[{i++}]";
                    Scope.Declare(var, this);

                    var comment = var.Comment;
                    if (Verbose)
                        comment = comment ?? var.ToString();
                    EmitAssign(var, expr, comment);
                    CheckDim(var);
                }
            }
        }

        public bool CompileExpr(IExpr expr, CodeGenerator generator)
        {
            if (!this.Scope.Contains(expr))
            {
                generator.Visit(expr, this);
//#if VERBOSE
//                if (this.Scope.Contains(expr)) CheckShape(expr, generator);
//#endif
            }
            return true;
        }

        public void CompileExpr(IEnumerable<IExpr> e, CodeGenerator generator)
        {
            foreach (var expr in e)
            {
                CompileExpr(expr, generator);
            }
        }

        private void EmitLine(string s = null, string comment = null)
        {
            if (s != null || comment != null)
                sb.Append('\t', this.Tab);
            if (s != null)
                sb.Append(s);
            if (s != null && comment != null)
                sb.Append("\t\t\t\t");
            if (comment != null)
            {
                sb.Append("// ");
                sb.Append(comment);
            }

            //sb.Append(Environment.NewLine);
            sb.Append('\n');		// avoid warning in VisualStudio
            ++CurrentLine;
        }

        public void EmitHeader()
        {
            this.EmitUsing("Proxem.NumNet");
            this.EmitUsing("Proxem.NumNet.Int32");
            this.EmitUsing("Proxem.NumNet.Single");
            this.EmitUsing("Proxem.TheaNet");
            this.EmitUsing("System");
            this.EmitUsing("System.Collections.Generic");
            this.EmitLine();
        }

        public void EmitStartBlock(string statement, string comment = null)
        {
            this.EmitLine(statement, comment);
            this.EmitLine("{");
            ++Tab;
            CreateScope();
        }

        public Scope CreateScope()
        {
            var scope = new Scope(this);
            this.Scope = scope;
            return scope;
        }

        public void EmitEndBlock()
        {
            --Tab;
            this.EmitLine("}");
            this.Scope = Scope.Parent;
        }

        public void EmitUsing(string package)
        {
            this.EmitLine("using " + package + ";");
        }

        public void EmitComment(string comment)
        {
            this.EmitLine(null, comment);
        }

        public void EmitAssign(IExpr var, string expr, string comment = null) =>
            this.EmitLine($"var {this.Scope.GetVar(var)} = {expr};", comment);

        public void EmitAliasing(IExpr var, IExpr expr, string comment = null) =>
            this.EmitLine($"{this.Scope.GetVar(var)} = {this.Scope.GetVar(expr)};", comment);

        public void EmitStore(Loop loop, int o, IExpr expr, string comment = null) =>
            this.EmitLine($"{this.Scope.GetVar(loop.Fors[o])}[i] = {this.Scope.GetVar(expr)};", comment);

        public void EmitReturn(IExpr output) => EmitLine($"return {this.Scope.GetVar(output)};");

        public void EmitReturn(IList<IExpr> outputs, Type outputsInAList)
        {
            if (outputsInAList == null && outputs.Count == 0)
                EmitLine("return;");
            else if (outputsInAList == null && outputs.Count == 1)
                EmitReturn(outputs[0]);
            else if (outputsInAList == null)
                EmitLine($"return Tuple.Create({string.Join(", ", outputs.Select(output => Scope.GetVar(output)))});");
            else
                EmitLine($"return new {outputsInAList.GetName()}[] {{ {string.Join(", ", outputs.Select(o => Scope.GetVar(o)))} }};");

            foreach (var output in outputs)
                DecCount(output);
        }

        public void EmitUpdates(OrderedDictionary updates, CodeGenerator generator)
        {
            if (updates != null)
            {
                // updates are emitted in 2 passes to allow swaps of variables:
                //      updates[a] = b;
                //      updates[b] = a;
                foreach (object key in updates.Keys)
                {
                    this.EmitComment($"Computing value needed for {key} update.");

                    this.CompileExpr((IExpr)updates[key], generator);
                    this.EmitLine();
                }

                foreach (object key in updates.Keys)
                    EmitUpdate((IExpr)key, (IExpr)updates[key], generator);
            }
        }

        public void EmitUpdate(IExpr shared, IExpr expr, CodeGenerator generator)
        {
            EmitComment("Updating " + shared);
            if (shared is ITensor && shared is IShared)
            {
                var tensor = (ITensor)expr;
                var buff = GetBuffer(tensor);
                if (buff != null && buff.IsShared && buff.Name == shared.Name && buff.References.Last() == expr)
                    EmitComment($"{Scope.GetVar(expr)} is already in {shared.Name}");
                else if (buff != null)
                    EmitLine($"Copy({Scope.GetVar(expr)}, result: {GetShared((IShared)shared)});");
                else
                    EmitLine($"{GetShared((IShared)shared)} = {Scope.GetVar(expr)};");
            }
            else if (shared is IShared)
                EmitLine($"{GetShared((IShared)shared)} = {Scope.GetVar(expr)};");
            else if (shared is Operators.Tensors.Slicing<float>)
            {
                var indexing = (Operators.Tensors.Slicing<float>)shared;
                this.CompileExpr((IExpr)indexing.Slices, generator);

                EmitLine($"{Scope.GetVar(indexing.x)}[{Scope.GetVar(indexing.Slices)}] = {Scope.GetVar(expr)};");
            }
            else
                EmitLine($"{InlineCodeGenerator.GetCode(shared)} = {Scope.GetVar(expr)};");
            DecCount(expr);
            EmitLine();
        }


        public string GetBufferName(BufferInfo buffer)
        {
            if (buffer == null)
                return null;
            else if (buffer.IsShared)
            {
                if (Scope.Contains(buffer.Name))
                    return buffer.Name;
                string shared;
                if (buffer.Content == typeof(NumNet.Array<int>))
                    shared = "IntArray";
                else if (buffer.Content == typeof(NumNet.Array<float>))
                    shared = "FloatArray";
                else
                    throw new NotImplementedException("Doesn't support shared of type: " + buffer.Content);
                return $"{shared}[\"{buffer.Name}\"]";
            }
            else if (buffer.IsResizable)
            {
                lockDecount = true;
                CompileExpr(buffer.Shape, new CodeGenerator());
                lockDecount = false;
                var shape = $"{string.Join(", ", buffer.Shape.Select(axis => Scope.GetVar(axis)))}";
                return buffer.Name + $".ResizeTo({shape})";
            }
            else
                return buffer.Name;
        }

        public void EmitBuffers()
        {
            if (mem == null) return;

            foreach (var buff in mem._assignables)
                EmitBuffer(buff);
        }

        public void EmitBuffer(BufferInfo buff)
        {
            if (buff.IsShared)
                return;

            var type = buff.Content.GetName();
            if (buff.IsResizable)
                EmitLine($"private static readonly ResizableArray<{type}> {buff.Name} = new ResizableArray<{type}>({string.Join(", ", buff.Shape.Select(d => d is Scalar<int>.Const ? d.ToString() : "-1"))});");
            else
                EmitLine($"private static readonly Array<{type}> {buff.Name} = NN.Zeros<{type}>({string.Join(", ", buff.Shape.Select(_ => _.ToString()))});");
        }

        public ReferenceCounter references = new ReferenceCounter();

        public int GetCount(IExpr expr) => references[expr];

        public void DecCount(IExpr expr)
        {
            if (!lockDecount)
                references.Decrement(expr);
        }

        public void Reference(IExpr expr) => references.Reference(expr);
        public void Dereference(IExpr expr) => references.Dereference(expr);

        public void Reference(IEnumerable<IExpr> e)
        {
            foreach (var expr in e)
            {
                Reference(expr);
            }
        }

        public void Reference(OrderedDictionary updates)
        {
            if (updates == null) return;
            foreach (IExpr expr in updates.Values)
            {
                Reference(expr);
            }
        }

        /// <summary>
        /// Returns the buffer allocated to the given tensor. Returns null if no buffer has been allocated.
        /// </summary>
        public BufferInfo GetBuffer(ITensor tensor) => mem != null && mem.Buffer.ContainsKey(tensor) ? mem.Buffer[tensor] : null;

        private void InitMem()
        {
            mem = mem ?? new MemoryAllocator(new Dictionary<IExpr, int>(references.Content));
        }

        /// <summary>
        /// Creates a mapping between expressions and buffers inside a function.
        /// </summary>
        public void CreateMemoryMapping(IEnumerable<IExpr> outputs, OrderedDictionary updates)
        {
            if (!Mem) return;

            InitMem();
            updates = updates ?? new OrderedDictionary();

            foreach (var target in updates.Keys)
                if (target is ITensor)
                {
                    // TODO: avoid the dynamic
                    dynamic shared = target;
                    CreateMemoryMapping(updates, shared);
                }

            foreach (var e in outputs)
                mem.Process(e);

            foreach (var expr in updates.Values)
                mem.Process((IExpr)expr);

            mem.Decrement(outputs);
            foreach (var expr in updates.Values)
                mem.Decrement((IExpr)expr);

            mem.AssertIsDone();
        }

        private void CreateMemoryMapping<T>(OrderedDictionary updates, Tensor<T>.Shared shared)
        {
            var expr = (Tensor<T>)updates[shared];
            var buff = BufferInfo.UseShared(shared);
            buff.References.Add(expr);
            mem.Buffer[expr] = buff;
        }

        public static string GetShared(IShared shared) => $"{GetSharedProperty(shared)}[\"{shared.Name}\"]";

        public static string GetSharedProperty(IShared shared)
        {
            // TODO: weak..., shall we add the underlying type of the shared as one of its member ?
            // shall we distinguih between int/Array<int>/float/Array<float> ? or just float/int (enough in combination with ITensor/IScalar)
            var type = shared.GetType().BaseType.BaseType;
            string name = type.GetName();
            switch (name)
            {
                case "Scalar<int>":
                    return "Int";
                case "Scalar<float>":
                    return "Float";
                case "Tensor<int>":
                    return "IntArray";
                case "Tensor<float>":
                    return "FloatArray";
            }
            return name;
        }

        /// <summary>
        /// Compile codes found in the private string builder.
        /// </summary>
        public FType GetMethod<FType>(string name) where FType : class
        {
            var source = GetSource();

            //var assembly = CompileWithCodeDom<FType>(source);
            var assembly = CompileWithRoslyn<FType>(source);

            var constructor = assembly.GetType("DynClass").GetConstructor(new Type[0]);
            var instance = (Runtime)constructor.Invoke(null);

            foreach (var function in CustomFunctions)
                instance.CustomFunctions.Add(function);

            var method = assembly.GetType("DynClass").GetMethod(name);

            var result = Delegate.CreateDelegate(typeof(FType), instance, method) as FType;
            if (result == null)
                throw new Exception("Can't convert compiled method to type " + typeof(FType).ToString());

            return result;
        }

        private Assembly CompileWithRoslyn<FType>(string source) where FType : class
        {
            var (sourceText, path) = GetSourceText(source, Debug);
            var syntaxTree = Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree.ParseText(sourceText, null, path);

            string assemblyName = Path.GetRandomFileName();
            var options = new Microsoft.CodeAnalysis.CSharp.CSharpCompilationOptions(
                Microsoft.CodeAnalysis.OutputKind.DynamicallyLinkedLibrary,
                optimizationLevel: Debug ? Microsoft.CodeAnalysis.OptimizationLevel.Debug : Microsoft.CodeAnalysis.OptimizationLevel.Release);

            var compilation = Microsoft.CodeAnalysis.CSharp.CSharpCompilation.Create(
                assemblyName,
                syntaxTrees: new[] { syntaxTree },
                references: Compiler.References,
                options: options);

            using (var peStream = new MemoryStream())
            using (MemoryStream pdbStream = Debug ? new MemoryStream() : null)
            {
                var result = compilation.Emit(peStream, pdbStream);

                if (!result.Success)
                {
                    var failures = result.Diagnostics.Where(diagnostic =>
                        diagnostic.IsWarningAsError ||
                        diagnostic.Severity == Microsoft.CodeAnalysis.DiagnosticSeverity.Error);

                    var sb = new StringBuilder();
                    sb.AppendLine($"Failed to compile method {nameof(FType)}.");
                    foreach (Microsoft.CodeAnalysis.Diagnostic diagnostic in failures)
                    {
                        sb.Append("CompilerError: ");
                        sb.AppendLine($"{diagnostic.Id}: {diagnostic.GetMessage()}");
                    }
                    sb.AppendLine("Source code:");
                    sb.AppendLine(source);
                    throw new Exception(sb.ToString());
                }
                return Assembly.Load(peStream.ToArray(), pdbStream?.ToArray());
            }
        }

        private static IEnumerable<string> GetTrustedPlatformAssemblies()
        {
            // see http://source.roslyn.io/#Microsoft.CodeAnalysis.Scripting/Hosting/Resolvers/RuntimeMetadataReferenceResolver.cs,180
            var type = Type.GetType("System.AppContext, System.AppContext, Version=4.1.0.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a");
            var getData = (Func<string, object>)Delegate.CreateDelegate(typeof(Func<string, object>), type.GetTypeInfo().GetDeclaredMethod("GetData"));

            if (getData.Invoke("TRUSTED_PLATFORM_ASSEMBLIES") is string paths)
            {
                foreach (var path in paths.Split(Path.PathSeparator))
                {
                    if (Path.GetExtension(path) == ".dll")
                    {
                        yield return path;
                    }
                }
            }
        }

        private static List<Microsoft.CodeAnalysis.MetadataReference> GetReferences()
        {
            var coreLib = CoreAssembly.Location;
            var references = new List<Microsoft.CodeAnalysis.MetadataReference>
            {
                Microsoft.CodeAnalysis.MetadataReference.CreateFromFile(coreLib),
                // "Proxem.NumNet.dll" full path
                Microsoft.CodeAnalysis.MetadataReference.CreateFromFile(typeof(Proxem.NumNet.Random).Assembly.Location),
                // "Proxem.TheaNet.dll" full path
                Microsoft.CodeAnalysis.MetadataReference.CreateFromFile(typeof(Compiler).Assembly.Location)
            };
            if (IsDotnetCore)
            {
                var coreDir = Path.GetDirectoryName(coreLib);
                references.Add(Microsoft.CodeAnalysis.MetadataReference.CreateFromFile(Path.Combine(coreDir, "System.Runtime.dll")));
                references.Add(Microsoft.CodeAnalysis.MetadataReference.CreateFromFile(Path.Combine(coreDir, "System.Console.dll")));
                foreach (var assembly in GetTrustedPlatformAssemblies().Where(assembly => Path.GetFileName(assembly) == "netstandard.dll"))
                {
                    references.Add(Microsoft.CodeAnalysis.MetadataReference.CreateFromFile(assembly));
                }
            }
            else
            {
                // "netstandard.dll" full path (NumNet uses netstandard 2.0)
                // Assumes that netstandard.dll is in same folder than current assembly.
                references.Add(Microsoft.CodeAnalysis.MetadataReference.CreateFromFile(
                Path.Combine(Path.GetDirectoryName(typeof(Compiler).Assembly.Location), "netstandard.dll")));
            }

            return references;
        }

        public static (Microsoft.CodeAnalysis.Text.SourceText, string) GetSourceText(string src, bool isDebug)
        {
            if (isDebug)
            {
                var path = Path.ChangeExtension(Path.GetTempFileName(), "cs");
                File.WriteAllText(path, src);
                using (var stream = File.OpenRead(path))
                {
                    return (Microsoft.CodeAnalysis.Text.SourceText.From(stream, Encoding.UTF8), path);
                }
            }
            else
            {
                return (Microsoft.CodeAnalysis.Text.SourceText.From(src), "");
            }
        }

        public static Assembly CoreAssembly = typeof(object).GetType().Assembly;
        public static bool IsDotnetCore = CoreAssembly.GetName().Name == "System.Private.CoreLib";
        public static List<Microsoft.CodeAnalysis.MetadataReference> References = GetReferences();

#if !NETSTANDARD
        private Assembly CompileWithCodeDom<FType>(string source) where FType : class
        {
            var references = new[] {
                // "Proxem.NumNet.dll" full path
                typeof(Proxem.NumNet.Random).Assembly.Location,
                // "Proxem.TheaNet.dll" full path
                this.GetType().Assembly.Location,
                // "netstandard.dll" full path (NumNet uses netstandard 2.0)
                // Assumes that netstandard.dll is in same folder than current assembly.
                Path.Combine(Path.GetDirectoryName(this.GetType().Assembly.Location), "netstandard.dll")
            };

            var provider = new CSharpCodeProvider();
            var cp = new System.CodeDom.Compiler.CompilerParameters();
            cp.GenerateExecutable = false;
            foreach (var reference in references)
                cp.ReferencedAssemblies.Add(reference);

            System.CodeDom.Compiler.CompilerResults cr;
            if (Compiler.Debug)
            {
                cp.IncludeDebugInformation = true;
                cp.GenerateInMemory = false;
                cp.TempFiles = new System.CodeDom.Compiler.TempFileCollection(Environment.GetEnvironmentVariable("TEMP"), true);
                var path = Path.ChangeExtension(Path.GetTempFileName(), "cs");
                //var path = "temp.cs";
                File.WriteAllText(path, source);    // for debugging purposes
                cr = provider.CompileAssemblyFromFile(cp, path);
            }
            else
            {
                cp.GenerateInMemory = true;
                cp.CompilerOptions = "/optimize";
                cr = provider.CompileAssemblyFromSource(cp, source);
            }

            if (cr.Errors.HasErrors)
            {
                var sb = new StringBuilder();
                sb.AppendLine($"Failed to compile method {nameof(FType)}.");
                foreach (var error in cr.Errors.Cast<System.CodeDom.Compiler.CompilerError>())
                {
                    sb.Append("CompilerError: ");
                    sb.AppendLine(error.ToString());
                }
                sb.AppendLine("Source code:");
                sb.AppendLine(source);
                throw new Exception(sb.ToString());
            }

            return cr.CompiledAssembly;
        }
#endif

        public void Assert(string v)
        {
            EmitLine("Assert(" + v + ");");
        }

        void CheckDim(IExpr expr)
        {
            var array = expr as ITensor;
            if (array == null) return;
            var var = expr as IVar;
            if (var == null || var.Name == null) return;
            EmitLine($"if ({Scope.GetVar(array)}.Shape.Length != {array.NDim}) throw new System.RankException(\"{var.Name}\");", $"({GetCount(array)})");
        }

        public void CheckShape(IExpr expr, CodeGenerator generator)
        {
            if (!Verbose)
                return;

            if (expr is ITensor array && array.NDim != 0)
            {
                CompileExpr(array.Shape, generator);
                EmitLine($"{Scope.GetVar(array)}.AssertOfShape({string.Join(", ", array.Shape.Select(axis => Scope.GetVar(axis)))});");
            }
        }

        public string OfShape(IExpr expr, CodeGenerator generator)
        {
            if (!Verbose)
                return "";

            if (expr is ITensor array && array.NDim != 0)
            {
                CompileExpr(array.Shape, generator);
                return $".OfShape({string.Join(", ", array.Shape.Select(axis => Scope.GetVar(axis)))})";
            }
            return "";
        }

        public string RefComment(IExpr left, params IExpr[] rights) => RefComment(left, rights, null);
        public string RefComment(IExpr left, IEnumerable<IExpr> rights, string comment = null)
        {
            var refComment = $"{GetCount(left)} = ({string.Join(", ", rights.Select(r => GetCount(r)))})";
            if (comment == null)
                return refComment;
            else
                return refComment + ": " + comment;
        }

        private static readonly IReadOnlyDictionary<string, int> precedences = new Dictionary<string, int>
        {
            ["Neg"] = 1,
            ["Div"] = 2,
            ["Mul"] = 2,
            ["Mod"] = 2,
            ["Add"] = 3,
            ["Sub"] = 3,
            ["Ge"] = 5,
            ["Gt"] = 5,
            ["Nq"] = 6,
        };

        public static int Precedences(IExpr expr)
        {
            if (expr.FunctionName == null)
                return 0;
            if (precedences.ContainsKey(expr.FunctionName))
                return precedences[expr.FunctionName];
            if (expr is Tensor<float>.Elementwise elementwise)
                return Precedences(elementwise.Abstraction);
            return 0;
        }
    }
}
