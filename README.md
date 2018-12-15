# TheaNet

TheaNet is an optimized library for deep learning models written in C\# and developed at [Proxem](https://proxem.com).
The library is inspired by python's [theano](http://deeplearning.net/software/theano/) library and offers similar possibilities.
Automatic gradient differentiation allows to create complex models and run them with simple commands with all the backpropagation being taken care of by the library.
TheaNet generates readable code in C\# with the possibility to step in the generated code and debug it.  

## Table of contents

* [Requirements](#requirements)
* [Nuget Package](#nuget-package)
* [Contact](#contact) 
* [License](#license)

## Requirements

TheaNet is currently developed in .Net Framework 4.7.2. 

TheaNet is integrated with our [NumNet](https://github.com/Proxem/NumNet) and [BlasNet](https://github.com/Proxem/BlasNet) libraries and require both libraries to work.  

## Nuget Package

We provide a Nuget Package of **TheaNet** to facilitate its use. It's available on [Nuget.org](https://www.nuget.org/packages/Proxem.TheaNet/). 
Symbols are also available to facilitate debugging inside the package.

## Debugging with TheaNet

![Define a scalar variable names "x"](https://github.com/Proxem/TheaNet/blob/master/images/debug1.png)

## Contact

If you can't make **TheaNet** work on your computer or if you have any tracks of improvement drop us an e-mail at one of the following address:
- thp@proxem.com
- joc@proxem.com

## License

TheaNet is Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
See the NOTICE file distributed with this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
