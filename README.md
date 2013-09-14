Summarization module (sentence extractor) based on Hidden Markov Model

All written in Go

Algorithms solving all three classic problems of HMM:
	0. Evaluation -> Forward algorithm
	1. Decoding -> Viterbi algorithm
	2. Optimization -> Baum-Welch algorithm

Main resource of code:
	http://www.codeproject.com/Articles/69647/Hidden-Markov-Models-in-C

Differences in my approach:
	0. Multiple features for emissions.
	1. Using concurrency for better performance.
	2. Integration with Redis DB (saving and loading model parameters).

Basic usage:
	> TODO
	Still raw printing result to standard input.