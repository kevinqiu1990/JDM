# JDM
A joint distribution matching (JDM) model for distribution-adaptation-based cross-project defect prediction.

JDM aims to minimize the joint distribution divergence between the source and target project to improve the CPDP performance. By constructing an adaptive weight vector for the instances of the source project, JDM can be effective and robust at reducing marginal distribution discrepancy and conditional distribution discrepancy simultaneously.

JDM.m is the implementation of JDM method.

Build running environment
=================

**1. Extract the version of _liblinear-weight_ according to the operating system (support Windows, Linux and MacOS). The liblinear-weight packages are located under {JDMroot}/tools/. Please ensure that only the proper one is unpacked.**

Liblinear is a simple package for solving large-scale regularized linear classification and regression. Liblinear-weight is a variant of liblinear that supports logical regression with instance weights.

**2. Download the _CVX package_ into {JDMroot}/tools/ from http://cvxr.com/cvx/download/. Please download the corresponding version to the operating system and follow the setup operation of CVX README.txt (Excuting cvs_setup command under root path of CVX).**

CVX is a Matlab-based modeling system for convex optimization. CVX turns Matlab into a modeling language, allowing constraints and objectives to be specified using standard Matlab expression syntax. Please note that under different operating systems, the results of CVX will be slightly different. The experimental environment of our paper is Windows 10, 64-bit, Intel Core 3.70 GHz server with 16GB RAM.

Demo
=================
After running environment building, please run demoJDM.m

Contacts
=================
If any issues, please feel free to contact the Author.

**Author Name**: Kevin Qiu

**Author Email**: qiushaojian@outlook.com