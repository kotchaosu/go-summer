# Go Summer

Summarization module (sentence extractor) based on Hidden Markov Model

100% Go

## Solutions for three classic problems of HMM:
	* Evaluation -> Forward algorithm
	* Decoding -> Viterbi algorithm
	* Optimization -> Baum-Welch algorithm

## Main resources:
	* http://www.codeproject.com/Articles/69647/Hidden-Markov-Models-in-C
	* http://disi.unitn.it/~passerini/teaching/complex_systems/slides/HMM.pdf

## What's new:
	* Multiple features for emissions.
	* Modified forward/backward/Viterbi for using "silent" states.
	* Integration with Redis DB (saving and loading model parameters).
	* Using concurrency for better performance (TODO).

## Basic usage:
	Still raw printing result to standard input.
