# Quantum Edward

Quantum Edward at this point is just a small library of Python tools for 
doing classical supervised learning on Quantum Neural Networks (QNNs). 

An analytical model of the QNN is entered as input into QEdward and the training
is done on a classical computer, using training data already available (e.g., 
MNIST), and using the famous BBVI (Black Box Variational Inference) method 
described in Reference 1 below. 

The input analytical model of the QNN is given as a sequence of gate 
operations for a gate model quantum computer. The hidden variables are 
angles by which the qubits are rotated. The observed variables are the input 
and output of the quantum circuit. Since it is already expressed in the qc's 
native language, once the QNN has been trained using QEdward, it can be 
run immediately on a physical gate model qc such as the ones that IBM and 
Google have already built. By running the QNN on a qc and doing 
classification with it, we can compare the performance in classification 
tasks of QNNs and classical artificial neural nets (ANNs). 

Other workers have proposed training a QNN on an actual physical qc. But 
current qc's are still fairly quantum noisy. Training an analytical QNN on a 
classical computer might yield better results than training it on a qc 
because in the first strategy, the qc's quantum noise does not degrade the 
training. 

The BBVI method is a mainstay of the "Edward" software library. Edward uses 
Google's TensorFlow lib to implement various inference methods (Monte Carlo 
and Variational ones) for Classical Bayesian Networks and for Hierarchical 
Models. H.M.s (pioneered by Andrew Gelman) are a subset of C.B. nets 
(pioneered by Judea Pearl). Edward is now officially a part of TensorFlow, 
and the original author of Edward, Dustin Tran, now works for Google. Before 
Edward came along, TensorFlow could only do networks with deterministic 
nodes. With the addition of Edward, TensorFlow now can do nets with both 
deterministic and non-deterministic (probabilistic) nodes. 

This first baby-step lib does not do distributed computing. The hope is that 
it can be used as a kindergarten to learn about these techniques, and that 
then the lessons learned can be used to write a library that does the same 
thing, classical supervised learning on QNNs, but in a distributed fashion 
using Edward/TensorFlow on the cloud. 

The first version of Quantum Edward analyzes two QNN models called NbTrols 
and NoNbTrols. These two models were chosen because they are interesting to 
the author, but the author attempted to make the library general enough so 
that it can accommodate other akin models in the future. The allowable 
models are referred to as QNNs because they consist of 'layers', 
as do classical ANNs (Artificial Neural Nets). TensorFlow can analyze 
layered models (e.g., ANN) or more general DAG (directed acyclic graph) 
models (e.g., Bayesian networks). 

This software is distributed under the MIT License.

References
----------

1. R. Ranganath, S. Gerrish, D. M. Blei, "Black Box Variational
Inference", https://arxiv.org/abs/1401.0118

2. https://en.wikipedia.org/wiki/Stochastic_approximation
discusses Robbins-Monro conditions

3. https://github.com/keyonvafa/logistic-reg-bbvi-blog/blob/master/log_reg_bbvi.py

4. http://edwardlib.org/

5. https://discourse.edwardlib.org/


