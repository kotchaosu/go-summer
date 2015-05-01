# Go Summer

Summarization (sentence extraction) module based on Hidden Markov Models. Written mostly in summer 2013.

## Solutions for three classic problems of HMM:

* Evaluation (forward algorithm)
* Decoding (Viterbi algorithm)
* Learning (counting)

## Main resources:

* Implementation fo Hidden Markov Models in C# [link](http://www.codeproject.com/Articles/69647/Hidden-Markov-Models-in-C).
* Presentation about Hidden Markov Models [link to PDF](http://disi.unitn.it/~passerini/teaching/complex_systems/slides/HMM.pdf).

## Depenedencies

* Redis client for Go [link](https://github.com/garyburd/redigo).

## What's new

* Multiple features for emissions.
* Integration with Redis DB (model parameters).
* Go concurrency features for better performance.

## Basic usage

Running learning process:

	$ ./summer <path to full textfiles> <path to texts summaries>

Summarizing a text:

	$ ./summer <path to textfile>

## In progress

* updating model with unsupervised learning (Baum-Welch algorithm)
* estimating emission distribution (functions instead of slices)
