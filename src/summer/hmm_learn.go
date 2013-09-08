// learning tools for HMM parameters
package main

import (
	"bufio"
	"fmt"
	"os"
	"nlptk"
	"math"
	"redis/redis"
)

const (
	FULL = "/home/maxabsent/Documents/learning_set/full_texts/"
	SUMM = "/home/maxabsent/Documents/learning_set/summarizations/"
	DBTRANSPREFIX = "#@#"
	DBEMISPREFIX = "@#@"
	DBDICTPREFIX = "#"
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

func Store(dictionary chan map[string]int) {
	connection := pool.Get()

	for d := range dictionary {
		for k, v := range d {

			k = DBDICTPREFIX + k

			_, err := connection.Do("EXISTS", k)
			if err != nil {
				connection.Do("SET", k, v) // create new entry
			} else {
				connection.Do("INCRBY", k, v) // update existing
			}
		}
	}
	connection.Close()
}

// function querying Redis dictionary
// return number of occurences of word in the training set
func GetWordCount(word string) int {
	connection := pool.Get()
	reply, err := redis.Values(connection.Do("GET", DBDICTPREFIX + word))
	connection.Close()
	
	if err != nil {
	    return nil
	} else {
		return reply
	}
}

// Builds vectors of features for every sentence
//
// Feature vector:
//	1. position within paragraph
//	2. number of terms in sentence
//	3. how likely the terms are given the baseline of terms
//	4. how likely the terms are given the document terms
func ComputeFeatures(sentence nlptk.Sentence, document_dict map[string]int) []float64 {
	
	all_words := float64(GetWordCount(DBDICTPREFIX + "TOTAL"))
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
	}http://www.youtube.com/watch?v=d4LdKapQaaE

	features := [...]float64{
		float64(number_of_sentence),
		float64(number_of_terms),
		baseline_likelihood,
		document_likelihood
	}
	return features
}


struct HiddenMM type {
	// number of states
	N int
	// number of observations per state
	M int
	// transition probabilities N x N
	A [][]float64
	// emission probabilities N x M
	B [][]float64
	// initial probability distribution vector
	Pi []float64
}

func (* HiddenMM) InitHMM(N, M int) {
	HiddenMM.Pi := make([]float64, N, N)
	HiddenMM.Pi[0] = 1.0

	HiddenMM.A := make([][]float64, N, N)
	for i := range HiddenMM.A {
		HiddenMM.A[i] = make([]float64, N, N)
	}

	HiddenMM.B := make([][]float64, N, N)
	for i := range HiddenMM.B {
		HiddenMM.B[i] = make([]float64, M, M)
	}
}

// Analyze full text and summarization to prepare observations:
//	- vectors of features
//	- binary table of sentence in summarization presence
func ObserveFile(filename string) [][]float64, []int {

	full, err := os.Open(FULL + filename)

	if err != nil {
		fmt.Println("Error reading file", FULL + filename)
		os.Exit(1)
	}

	summ, err := os.Open(SUMM + filename)

	if err != nil {
		fmt.Println("Error reading file", SUMM + filename)
		os.Exit(1)
	}

	observations := make([][]float64, 10, 10)
	sentence_counter := make([]int, 10, 10)
	sentence_number := 0
	paragraph_number := 0

	reader_summ := bufio.NewReader(summ)
	spar, e := reader_summ.ReadBytes('\n')
	summarization := nlptk.Paragraph{paragraph_number, string(spar)}
	sum_sentences := summarization.GetParts()

	reader_full := bufio.NewReader(full)
	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
		paragraph := nlptk.Paragraph{paragraph_number, string(bpar)}
		sentences := paragraph.GetParts()
		
		if len(sentences) == 0 {
			continue
		}

		for i, s := range sentences {
			observations[sentence_number] = make([]float64, 4, 4)
			observations[sentence_number] = ComputeFeatures(nlptk.Sentence{i, s}, WordCount(FULL + filename))

			sentence_counter[sentence_number] = 0

			for _, sum := range sum_sentences {
				if s == sum {
					sentence_counter[sentence_number] = 1
					break
				}
			}			
			sentence_number++
			ExtendSlice(observations, sentence_number)
			ExtendSlice(sentence_counter, sentence_number)
		}
	}
	return observations, sentence_counter
}

func ExtendSlice(slice []interface{}, limit int) {
	if limit == len(slice) - 1 {
		temp := make([]int, len(slice)+10, len(slice)+10)
		copy(temp, slice)
		slice = temp
	}
}

//
func CheckConvergence(old_likelihood float64, new_likelihood float64,
	current_iter int, max_iter int, tolerance float64) bool {

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
func (* HiddenMM) Forward(observations []float64, *c []float64) [][]float64 {
	T := len(observations)
	pi, A, B := HiddenMM.Pi, HiddenMM.A, HiddenMM.B
	fwd := make([][]float64, T)

	for i := range fwd {
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
func (* HiddenMM) Backward(observations []float64, c []float64) [][]float64 {
	T := len(observations)
	pi, A, B := HiddenMM.Pi, HiddenMM.A, HiddenMM.B
	bwd := make([][]float64, T)

	for i := range bwd {
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
func (* HiddenMM) Learn(observations [][]float64, iterations int, tolerance float64) {
	if tolerance * iterations == 0 {
		return
	}

	N := len(observations)
	iter := 0
	stop := false

	pi := make([]float64, N)  // probabilities
	A := make([][]float64, N)  // transitions
	B := make([][]float64, )  // emissions

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

	for _, f := range files_slice {
		InitHMM(f)
	}

	fmt.Println("Learning succeed!")
}
