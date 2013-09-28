# Go Summer

Summarization module (sentence extractor) based on Hidden Markov Model

100% Go

## Solutions for three classic problems of HMM:
	* Evaluation -> Forward algorithm
	* Decoding -> Viterbi algorithm
	* Learning -> counting (todo for unsupervised learning: Baum-Welch algorithm)

## Main resources:
	* http://www.codeproject.com/Articles/69647/Hidden-Markov-Models-in-C
	* https://github.com/garyburd/redigo
	* http://disi.unitn.it/~passerini/teaching/complex_systems/slides/HMM.pdf

## What's new:
	* Multiple features for emissions.
	* Integration with Redis DB (saving and loading model parameters).
	* Using concurrency for better performance.

## Basic usage:
	
	command: ./summer filepath -> summarize text file
			 ./summer filepath full_set_path summ_set_path -> learn from data and summarize

## Actually improved:
	* updating model with unsupervised learning
	* preparing web app