// learning tools for HMM parameters
package main

import (
	"bufio"
	"fmt"
	"os"
	"nlptk"
	"math"
	"dbconn"
	"redis/redis"
)

const (
	FULL = "/home/maxabsent/Documents/learning_set/full_texts/"
	SUMM = "/home/maxabsent/Documents/learning_set/summarizations/"
	DBTRANSPREFIX = "#@#"
	DBEMISPREFIX = "@#@"
	DBDICTPREFIX = "#"
	NSTATES = 480
	NVALS = 1
)


// function querying Redis dictionary
// return number of occurences of word in the training set
func GetWordCount(word string) int {
	connection := dbconn.Pool.Get()
	reply, _ := redis.Int(connection.Do("GET", DBDICTPREFIX + word))
	connection.Close()
	
	return reply
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
	}

	features := []float64{
		float64(number_of_sentence),
		float64(number_of_terms),
		baseline_likelihood,
		document_likelihood,
	}
	return features
}


type HiddenMM struct {
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


func InitHMM(N, M int) HiddenMM {
	fmt.Println("Creating model")

	Pi := make([]float64, N, N)
	
	for i := range Pi {
		Pi[i] = 1.0 / float64(N)
	}

	A := make([][]float64, N, N)
	for i := range A {
		A[i] = make([]float64, N, N)

		for j := range A[i] {
			A[i][j] = 1.0 / float64(N)
		}
	}

	B := make([][]float64, N, N)
	for i := range B {
		B[i] = make([]float64, M, M)

		for j := range B[i] {
			B[i][j] = 1.0 / float64(M)
		}
	}

	fmt.Println("A", len(A), len(A[0]))
	fmt.Println("B", len(B), len(B[0]))
	fmt.Println("Pi", len(Pi))

	return HiddenMM{N, M, A, B, Pi}
}


// Analyze full text and summarization to prepare observations:
//	- vectors of features
//	- binary table of sentence in summarization presence
func ObserveFile(filename string) ([][]float64, []int) {

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

	observations := make([][]float64, NSTATES, NSTATES)
	sentence_counter := make([]int, NSTATES, NSTATES)
	sentence_number := 0
	paragraph_number := 0

	reader_summ := bufio.NewReader(summ)
	spar, _ := reader_summ.ReadBytes('\n')
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
			observations[sentence_number] = ComputeFeatures(nlptk.Sentence{i, s}, nlptk.WordCount(FULL + filename))

			not_in_summ := true
			for _, sum := range sum_sentences {
				if s == sum {
					not_in_summ = false
					sentence_counter[2 * sentence_number + 1] = 1
					break
				}
			}
			if not_in_summ {
				sentence_counter[2 * sentence_number] = 1
			}
			sentence_number++
		}
	}

	fmt.Println("sequence", filename, len(sentence_counter), sentence_counter)

	return observations, sentence_counter
}


func CheckConvergence(old_likelihood float64, new_likelihood float64, current_iter int, max_iter int, tolerance float64) bool {
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
	}
	return false
}


// Kalman filter
// Calculate probability of generated sequence
// http://en.wikipedia.org/wiki/Forward_algorithm
// func (* HiddenMM) Forward(observations [][]float64, c []float64) [][]float64 {
func (h *HiddenMM) Forward(observation []int, c []float64) [][]float64 {
	fwd := make([][]float64, len(observation))

	for i := range fwd {
		fwd[i] = make([]float64, h.N)
	}

	// STEP 1
	// init
	for i := 0; i < h.N; i++ {
		fwd[0][i] = h.Pi[i] * h.B[i][observation[0]]
		c[0] += fwd[0][i]
	}

	// scaling
	if c[0] > 0 {
		for i := 0; i < h.N; i++ {
			fwd[0][i] /= c[0]
		}
	}

	fmt.Println("fwd", len(fwd), len(fwd[0]))
	fmt.Println("c", len(c))

	// STEP 2
	// induction
	for t := 1; t < len(observation); t++ {
		for i := 0; i < h.N; i++ {
			p := h.B[i][observation[t]]

			sum := 0.0

			for j := 0; j < h.N; j++ {
				sum += fwd[t - 1][j] * h.A[j][i]
			}
			fwd[t][i] = sum * p

			c[t] += fwd[t][i]
		}

		// scaling
		if c[t] > 0 {
			for i := 0; i < h.N; i++ {
				fwd[t][i] /= c[t]
				fmt.Println("forward c[t]", c[t])
			}
		}
	}
	fmt.Println("forward scaling", c)

	return fwd
}


// Kalman smoothing
// Backward variables - use the same 'forward' scaling factor
// func (* HiddenMM) Backward(observations []float64, c []float64) [][]float64 {
func (h *HiddenMM) Backward(observations []int, c []float64) [][]float64 {
	T := len(observations)
	bwd := make([][]float64, T)

	for i := range bwd {
		bwd[i] = make([]float64, h.N)
	}
	// STEP 1
	// init
	for i := 0; i < h.N; i++ {
		bwd[T - 1][i] = 1.0 / c[T - 1]
	}

	// STEP 2
	// induction
	for t := T - 2; t >= 0; t-- {
		for i := 0; i < h.N; i++ {
			sum := 0.0
			for j := 0; j < h.N; j++ {
				sum += h.A[i][j] * h.B[j][observations[t + 1]] * bwd[t + 1][j]
			}
			bwd[t][i] += sum / c[t]
		}
	}
	return bwd
}


// main algorithm for learning HMM parameters from given observations
func (h *HiddenMM) Learn(observations [][]int, iterations int, tolerance float64) float64 {
	if tolerance * float64(iterations) == 0.0 {
		return 0.0
	}
	iter := 0
	stop := false

	// init
	epsilon := make([][][][]float64, len(observations))
	gamma := make([][][]float64, len(observations))

	for i := 0; i < len(observations); i++ {
		epsilon[i] = make([][][]float64, len(observations[i]))
		gamma[i] = make([][]float64, len(observations[i]))

		for j := 0; j < len(observations[i]); j++ {
			epsilon[i][j] = make([][]float64, h.N)
			gamma[i][j] = make([]float64, h.N)

			for k := 0; k < h.N; k++ {
				epsilon[i][j][k] = make([]float64, h.N)
			}
		}
	}

	// initial log-likelihood
	old_likelihood := 0.00001
	new_likelihood := 0.0

	// maintain loop until reaching tolerance
	// or number of iterations
	for !stop {
		// for each sequence of observations
		for i := 0; i < len(observations); i++ {
			scaling := make([]float64, h.N)
			
			fmt.Println("A", h.A[0], "new_likelihood", new_likelihood)
			fmt.Println("#", scaling)

			// I STEP
			// calculate forward & backward probability
			fwd := h.Forward(observations[i], scaling)
			bwd := h.Backward(observations[i], scaling)

			// II STEP
			// transmission & emission pairs

			// gamma
			for j := 0; j < len(observations[i]); j++ {
				s := 0.0

				for k := 0; k < h.N; k++ {
					gamma[i][j][k] = fwd[j][k] * bwd[j][k]
					s += gamma[i][j][k]
				}

				// scaling
				if s > 0 {
					for k := 0; k < h.N; k++ {
						gamma[i][j][k] /= s
					}
				}
			}

			// epsilon
			for j := 0; j < len(observations[i]) - 1; j++ {
				s := 0.0

				for k := 0; k < h.N; k++ {
					for l := 0; l < h.N; l++ {
						epsilon[i][j][k][l] = fwd[j][k] * h.A[k][l] * bwd[j + 1][l] * h.B[l][observations[i][j + 1]]
						s += epsilon[i][j][k][l]
					}
				}

				if s > 0 {
					for k := 0; k < h.N; k++ {
						for l := 0; l < h.N; l++ {
							epsilon[i][j][k][l] /= s
						}
					}
				}
			}

			// log-likelihood for the sequence
			for t := 0; t < len(scaling); t++ {
				new_likelihood += math.Log(scaling[t])
			}
			fmt.Println("new_likelihood for sequence", new_likelihood)
		}

		// average likelihood
		fmt.Println("A1", new_likelihood)
		new_likelihood /= float64(len(observations))
		fmt.Println("A2", new_likelihood)

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
			for k := 0; k < h.N; k++ {
				sum := 0.0
				for i := 0; i < len(observations); i++ {
					sum += gamma[i][0][k]
				}
				h.Pi[k] = sum / float64(h.N)
			}

			// transition probabilities
			for i := 0; i < h.N; i++ {
				for j := 0; j < h.N; j++ {
					den, num := 0.0, 0.0
					for k := 0; k < len(observations); k++ {
						T := len(observations[k])

						for l := 0; l < T - 1; l++ {
							num += epsilon[k][l][i][j]
						}
						for l := 0; l < T - 1; l++ {
							den += gamma[k][l][i]
						}
					}

					if den == 0.0 {
						h.A[i][j] = 0.0
					} else {
						h.A[i][j] = num / den
					}
				}
			}

			// emission probabilities
			for i := 0; i < h.N; i++ {
				for j := 0; j < h.M; j++ {
					den, num := 0.0, 0.0
					for k := 0; k < len(observations); k++ {
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
						h.B[i][j] = 0.0
					} else {
						h.B[i][j] = num / den
					}
				}
			}
		}
	}
	// return the model avg log-likelihood
	return new_likelihood
}


func (h *HiddenMM) Store() {
	fmt.Println("A")
	for i, v := range h.A {
		fmt.Println(i, v)
	}

	fmt.Println("B")
	for i, v := range h.B {
		fmt.Println(i, v)
	}
}


func main() {
	dir, err := os.Open(FULL)

	if err != nil {
		fmt.Println("Error reading directory", FULL)
		os.Exit(1)
	}

	// select all filenames from open directory
	files_slice, err := dir.Readdirnames(0)

	if err != nil {
		fmt.Println("Error reading filenames from directory", FULL)
		os.Exit(1)
	}	

	err = dir.Close()

	if err != nil {
		fmt.Println("Error closing directory", FULL)
		os.Exit(1)
	}

	hmm := InitHMM(NSTATES, NVALS)

	// fmt.Println("A PRE", hmm.A)

	// prepare learning set
	// for every input file
	//   for every state in file
	//     for every observed symbol
	learning_set := make([][][]float64, len(files_slice))
	observed_sequence := make([][]int, len(files_slice))

	for i, f := range files_slice {
		fmt.Println("Processing", f, "file")
		learning_set[i], observed_sequence[i] = ObserveFile(f)
	}

	fmt.Println("Begin learning...")
	hmm.Learn(observed_sequence, len(observed_sequence), 0.01)
	
	fmt.Println("Saving model in database...")
	hmm.Store()

	fmt.Println("Learning succeed!")
}
