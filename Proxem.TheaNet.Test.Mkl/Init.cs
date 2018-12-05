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
using System.Configuration;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Proxem.BlasNet;

namespace Proxem.TheaNet.Test.Mkl
{
    [TestClass]
    public class Init
    {
        [AssemblyInitialize]
        public static void InitProvider(TestContext context)
        {
            var path = ConfigurationManager.AppSettings["mkl:Path"];
            var threads = int.Parse(ConfigurationManager.AppSettings["mkl:Threads"] ?? "-1");

            if (!Directory.Exists(path))
                throw new DirectoryNotFoundException($"The MKL libs directory '{path}' was not found. Check appSetting 'mkl:Path'.");

            StartProvider.LaunchMklRt(threads, path);
        }
    }
}
