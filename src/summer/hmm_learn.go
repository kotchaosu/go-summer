// learning tools for HMM parameters

// package extracts and counts all words from learning set;
// output (map[string]int) is saved in Redis DB
package main

import (
	"bufio"
	"fmt"
	"os"
	"nlptk"
	"math"
	"redis"
)

const (
	FULL = "/home/maxabsent/Documents/learning_set/full_texts/"
	SUMM = "/home/maxabsent/Documents/learning_set/summarizations/"
)

// extracts, trims from special signs and counts "bare" words in learning set
func WordCount(file_path string) map[string]int {
	file, err := os.Open(file_path)

	if err != nil {
		fmt.Println("Error reading file", file_path)
		os.Exit(1)
	}

	reader := bufio.NewReader(file)
	word_counter := make(map[string]int)

	for bpar, e := reader.ReadBytes('\n'); e == nil; bpar, e = reader.ReadBytes('\n') {
		paragraph := nlptk.Paragraph{0, string(bpar)}
		sentences := paragraph.GetParts()
		
		for _, sentence := range sentences {
			s := nlptk.Sentence{0, sentence}
			words := s.GetParts()

			if len(words) == 0 {
				continue
			}

			for _, word := range words {
				word_counter[word]++
				word_counter["TOTAL"]++
			}
		}
	}
	return word_counter
}

func Store(dictionary map[string]int) {
	// connect to db
	connection := redis.newConnHdl(redis.DefaultSpec())
	connection.connect()
	// for every element in input dictionary
	// check if word exists in db
	//		1. create new row
	//		2. update the old one
	// close connection
	connection.disconnect()
}

// function querying Redis dictionary
// return number of occurences of word in the training set
func GetWordCount(word string) int {
	//
}

// builds vectors of features for every sentence
// feature vector:
//	1. position within paragraph
//	2. number of terms in sentence
//	3. how likely the terms are given the baseline of terms
//	4. how likely the terms are given the document terms
func ComputeFeatures(sentence nlptk.Sentence, document_dict map[string]int) []float64 {
	
	all_words := float64(GetWordCount("TOTAL"))
	all_doc_words := float64(document_dict["TOTAL"])	

	words := sentence.GetParts()

	number_of_sentence := float64(sentence.Number)
	number_of_terms := float64(len(words))
	baseline_likelihood := 0.0
	document_likelihood := 0.0

	for _, word := range words {
		base_count := float64(GetWordCount(word))
		doc_count := float64(document_dict[word])

		baseline_likelihood += math.Log10(base_count/all_words)
		document_likelihood += math.Log10(doc_count/all_doc_words)
	}

	features := [...]float64{
		float64(number_of_sentence),
		float64(number_of_terms),
		baseline_likelihood,
		document_likelihood
	}
	return features
}

// process full text and human-created summarization
// to initialize Hidden Markov Model of 2s+1 states
// (s - number of sentences in full text)
// with probabilitiy of the sentence occurrence in the summary
func InitHMM(filename string) [][]float64 {
	// open two files
	// find and write down number of sentence which appeared in the summary
}

//
func CheckConvergence(old_likelihood float64, new_likelihood float64,
	current_iter int, max_iter int, tolerance float64) bool {
	//
	if tolerance > 0 {
		if math.Abs(old_likelihood - new_likelihood) <= tolerance {
			return true
		}

		if max_iter > 0 {
			if current_iter >= max_iter {
				return true
			}
		}
	} else {
		if current_iter == max_iter {
			return true
		}
		return false
	}
}

// calculate probability of generated sequence
func Forward(observations []float64, *c []float64) [][]float64 {
	T := len(observations)
	pi := make([]float64, N)
	A := make([][]float64, N)

	fwd := make([][]float64, T)

	for i := range(fwd) {
		fwd[i] = make([]float64, States)
	}
	c := make([]float64, T)

	// STEP 1
	// init
	for i := 0; i < States; i++ {
		fwd[0][i] = pi[i] * B[i][observations[0]]
		c[0] += fwd[0][i]
	}

	// scaling
	if c[0] > 0 {
		for i := 0; i < States; i++ {
			fwd[0][i] /= c[0]
		}
	}

	// STEP 2
	// induction
	for t := 1; t < T; t++ {
		for i := 0; i < States; i++ {
			p := B[i][observations[t]]

			sum := 0.0

			for j := 0; j < States; j++ {
				sum += fwd[t - 1][j] * A[j][i]
			}
			fwd[t][i] = sum * p

			c[t] += fwd[t][i]
		}

		// scaling
		if c[t] > 0 {
			for i := 0; i < States; i++ {
				fwd[t][i] /= c[t]
			}
		}
	}
	return fwd
}

// backward variables - use the same 'forward' scaling factor
func Backward(observations []float64, c []float64) [][]float64 {
	T := len(observations)
	pi := make([]float64, N)
	A := make([][]float64, N)

	bwd := make([][]float64, T)

	for i := range(bwd) {
		bwd[i] = make([]float64, States)
	}
	// STEP 1
	// init
	for i := 0; i < States; i++ {
		bwd[T - 1][i] = 1.0 / c[T - 1]
	}

	// STEP 2
	// induction
	for t := T - 2; t >= 0; t-- {
		for i := 0; i < States; i++ {
			sum := 0.0
			for j := 0; j < States; j++ {
				sum += A[i][j] * B[j][observations[t + 1]] * bwd[t + 1][j]
			}
			bwd[t][i] += sum / c[t]
		}
	}
	return bwd
}

// main algorithm for learning HMM parameters from given observations
func Learn(observations [][]float64, iterations int, tolerance float64) {
	if tolerance * iterations == 0 {
		return
	}

	N := len(observations)
	iter := 0
	stop := false

	pi := make([]float64, N)  // probabilities
	A := make([][]float64, N)  // transitions

	// init
	epsilon := make([][]float64, N)
	gamma := make([][]float64, N)

	for i := 0; i < N; i++ {
		T := len(observations[i])
		epsilon[i] = [...]float64{T, States, States}
		gamma[i] = [...]float64{T, States}
	}

	// initial log-likelihood
	old_likelihood := 0.00001
	new_likelihood := 0.0

	// maintain loop until reaching tolerance
	// or number of iterations
	for !stop {

		// for each sequence of observations
		for i := 0; i < N; i++ {
			sequence := observations[i]
			T := len(sequence)

			scaling := make([]float64)

			// I STEP
			// calculate forward & backward probability
			fwd := Forward(observations[i], &scaling)
			bwd := Backward(observations[i], &scaling)

			// II STEP
			// transmission & emission pairs

			// gamma
			for t := 0; t < T; t++ {
				s := 0.0

				for k := 0; k < States; k++ {
					gamma[i][t][k] = fwd[t][k] * bwd[t][k]
					s += gamma[i][t][k]
				}

				// scaling
				if s > 0 {
					for k := 0; k < States; k++ {
						gamma[i][t][k] /= s
					}
				}
			}

			// epsilon
			for t := 0; t < T - 1; t++ {
				s := 0.0

				for k := 0; k < States; k++ {
					for l := 0; l < States; l++ {
						epsilon[i][t][k][l] = fwd[t][k] * A[k][l] * bwd[t + 1][l] * B[l][sequence[t + 1]]
						s += epsilon[i][t][k][l]
					}
				}

				if s > 0 {
					for k := 0; k < States; k++ {
						for l := 0; l < States; l++ {
							epsilon[i][t][k][l] /= s
						}
					}
				}
			}

			// log-likelihood for the sequence
			for t := 0; t < len(scaling); t++ {
				new_likelihood += math.Log(scaling[t])
			}
		}

		// average likelihood
		new_likelihood /= len(observations)

		// check convergence
		if CheckConvergence(old_likelihood, new_likelihood, iter, iterations, tolerance) {
			stop = true
		} else {
			// STEP 3
			// parameter re-estimation
			iter++
			old_likelihood = new_likelihood
			new_likelihood = 0.0

			// init probabilities
			for k := 0; k < States; k++ {
				sum := 0.0
				for i := 0; i < N; i++ {
					sum += gamma[i][0][k]
				}
				pi[k] = sum / N
			}

			// transition probabilities
			for i := 0; i < States; i++ {
				for j := 0; j < States; j++ {
					den, num := 0.0, 0.0
					for k := 0; k < N; k++ {
						T := len(observations[k])

						for l := 0; l < T - 1; l++ {
							num += epsilon[k][l][i][j]
						}
						for l := 0; l < T - 1; l++ {
							den += gamma[k][l][i]
						}
					}

					if den == 0.0 {
						A[i][j] = 0.0
					} else {
						A[i][j] = num / den
					}
				}
			}

			// emission probabilities
			for i := 0; i < States; i++ {
				for j := 0; j < States; j++ {
					den, num := 0.0, 0.0
					for k := 0; k < N; k++ {
						T := len(observations[k])
						for l := 0; l < T; l++ {
							if observations[k][l] == j {
								num += gamma[k][l][i]
							}
						}
						for l := 0; l < T; l++ {
							den += gamma[k][l][i]
						}
					}

					if den == 0.0 {
						B[i][j] = 0.0
					} else {
						B[i][j] = num / den
					}
				}
			}
		}
	}
	// return the model avg log-likelihood
	return new_likelihood
}

func main() {
	dir, err := os.Open(SETDIR)

	if err != nil {
		fmt.Println("Error reading directory", SETDIR)
		os.Exit(1)
	}

	// select all filenames from open directory
	files_slice, err := dir.Readdirnames(0)

	if err != nil {
		fmt.Println("Error reading filenames from directory", SETDIR)
		os.Exit(1)
	}	

	err := dir.Close()

	if err != nil {
		fmt.Println("Error closing directory", SETDIR)
		os.Exit(1)
	}

	// count words from all files
	// and update database dictionary
	c := make(chan map[string]int)
	
	go func() {
		for _, v := range(files_slice) {
			c <- WordCount(SETDIR + v)
		}
		Store(<-c)
	}()

	fmt.Println("Learning succeed!")
}
