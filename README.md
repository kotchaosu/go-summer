# Go Summer

Summarization module (sentence extractor) based on Hidden Markov Model

All written in Go

## Algorithms solving all three classic problems of HMM:
	* Evaluation -> Forward algorithm
	* Decoding -> Viterbi algorithm
	* Optimization -> Baum-Welch algorithm

## Main resource of code:
	http://www.codeproject.com/Articles/69647/Hidden-Markov-Models-in-C

## Differences in my approach:
	* Multiple features for emissions.
	* Modified forward/backward/Viterbi for using "silent" states.
	* Integration with Redis DB (saving and loading model parameters).
	* Using concurrency for better performance (TODO).

## Basic usage:
	> TODO
	Still raw printing result to standard input.