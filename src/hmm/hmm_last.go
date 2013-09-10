// For optimising runtime
// Assuming only two possible transitions for each states
// 1 - summary, 2 - non-summary

// learning tools for HMM parameters
package hmm

import (
	"fmt"
	"math"
	"bufio"
	"os"
	"strings"
	"nlptk"
	"dbconn"
)

const (
	FULL = "/home/maxabsent/Documents/learning_set/full_texts/"
	SUMM = "/home/maxabsent/Documents/learning_set/summarizations/"
	
	DBA = "@@#"
	DBB = "@#@"
	DBPI = "@@@"
	
	DBDICTPREFIX = "#"
	
	NSTATES = 300
	NVALS = 300
)


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
	sentence_counter := make([]int, 0, 0)
	sentence_number := 0
	paragraph_number := 0

	reader_summ := bufio.NewReader(summ)
	spar, _ := reader_summ.ReadBytes('\n')
	summarization := nlptk.Paragraph{paragraph_number, string(spar)}
	sum_sentences := summarization.Text

	reader_full := bufio.NewReader(full)
	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
		paragraph := nlptk.Paragraph{paragraph_number, string(bpar)}
		sentences := paragraph.GetParts()

		if len(sentences) == 0 {
			continue
		}

		for _, s := range sentences {
			observations[sentence_number] = make([]float64, 4, 4)
			// observations[sentence_number] = float64(i)  // ComputeFeatures(nlptk.Sentence{i, s}, nlptk.WordCount(FULL + filename))

			if strings.Contains(sum_sentences, s) {
				sentence_counter = append(sentence_counter, 2 * sentence_number + 1)
			} else {
				sentence_counter = append(sentence_counter, 2 * sentence_number)
			}
			sentence_number++

			// safety switch
			if sentence_number == NSTATES {
				return observations, sentence_counter
			}
		}
	}

	fmt.Println("sequence", filename, len(sentence_counter), sentence_counter)

	return observations, sentence_counter
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
	Pi[0] = 1.0

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

	return HiddenMM{N, M, A, B, Pi}
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

	if math.IsNaN(new_likelihood) || math.IsInf(new_likelihood, 0) {
		return true
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
	if c[0] != 0 {
		for i := 0; i < h.N; i++ {
			fwd[0][i] = fwd[0][i] / c[0]
		}
	}

	// STEP 2
	// induction
	for t := 1; t < len(observation); t++ {
		for i := 0; i < h.N; i++ {
			for j := 0; j < h.N; j++ {
				p := h.A[j][i] * h.B[i][observation[t]]
				fwd[t][i] += fwd[t - 1][j] * p
			}

			c[t] += fwd[t][i]  // likelihood
		}

		// scaling
		if c[t] != 0 {
			for i := 0; i < h.N; i++ {
				fwd[t][i] = fwd[t][i] / c[t]
			}
		}
	}

	return fwd
}


// Kalman smoothing
// Backward variables - use the same 'forward' scaling factor
// func (* HiddenMM) Backward(observations []float64, c []float64) [][]float64 {
func (h *HiddenMM) Backward(observation []int, c []float64) [][]float64 {
	T := len(observation)
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
				sum += h.A[i][j] * h.B[j][observation[t + 1]] * bwd[t + 1][j]
			}
			bwd[t][i] = bwd[t][i] + sum / c[t]
		}
	}
	return bwd
}


// main algorithm for learning HMM parameters from given observations
func (h *HiddenMM) Learn(observations [][]int, iterations int, tolerance float64) float64 {
	if tolerance + float64(iterations) == 0.0 {
		return 0.0
	}
	iter := 1
	stop := false

	// init
	epsilon := make([][][][]float64, len(observations))
	gamma := make([][][]float64, len(observations))

	for i := range observations {
		epsilon[i] = make([][][]float64, len(observations[i]))
		gamma[i] = make([][]float64, len(observations[i]))

		for j := range observations[i] {
			epsilon[i][j] = make([][]float64, h.N)
			gamma[i][j] = make([]float64, h.N)

			for k := 0; k < h.N; k++ {
				epsilon[i][j][k] = make([]float64, h.N)
			}
		}
	}

	// initial log-likelihood
	old_likelihood := math.SmallestNonzeroFloat64
	new_likelihood := float64(0)

	// maintain loop until reaching tolerance
	// or number of iterations
	for !stop {
		// for each sequence of observations
		for i := range observations {
			scaling := make([]float64, len(observations[i]))

			// I STEP
			// calculate forward & backward probability
			fwd := h.Forward(observations[i], scaling)
			bwd := h.Backward(observations[i], scaling)

			// II STEP
			// transmission & emission pairs

			// gamma
			for j := range observations[i] {
				s := float64(0)

				for k := 0; k < h.N; k++ {
					gamma[i][j][k] = fwd[j][k] * bwd[j][k]
					s += gamma[i][j][k]
				}

				// scaling
				if s != 0 {
					for k := 0; k < h.N; k++ {
						gamma[i][j][k] = gamma[i][j][k] / s
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

				if s != 0 {
					for k := 0; k < h.N; k++ {
						for l := 0; l < h.N; l++ {
							epsilon[i][j][k][l] = epsilon[i][j][k][l] / s
						}
					}
				}
			}
			// log-likelihood for the sequence
			for t := range scaling {
				new_likelihood += math.Log(scaling[t])
			}
		}

		// average likelihood
		new_likelihood /= float64(len(observations))

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
				for i := range observations {
					sum += gamma[i][0][k]
				}
				h.Pi[k] = sum / float64(len(observations))
			}

			// transition probabilities
			for i := 0; i < h.N; i++ {
				for j := 0; j < h.N; j++ {
					den, num := 0.0, 0.0
					for k := range observations {
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
					for k := range observations {
						for l := range observations[k] {
							if observations[k][l] == j {
								num += gamma[k][l][i]
							}
						}
						for l := range observations[k] {
							den += gamma[k][l][i]
						}
					}

					if num == 0.0 {
						h.B[i][j] = 1e-10
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


func (h *HiddenMM) Evaluate(observation []int, logarithm bool) float64 {
    // Forward algorithm
    likelihood := float64(0);
    coefficients := make([]float64, len(observation))

    // Compute forward probabilities
    h.Forward(observation, coefficients)

    for _, v := range coefficients {
    	likelihood += math.Log(v);
    }
    // Return the sequence probability
    if logarithm {
    	return likelihood
    }
    return math.Exp(likelihood)
}


func (h *HiddenMM) Viterbi(observation []int, probability *float64) []int {
	
	T := len(observation)
	min_state := 0
	min_weight := 0.0

	s := make([][]int, h.N)
	for i := range s {
		s[i] = make([]int, T)
	}	

	a := make([][]float64, h.N)
	for i := range a {
		a[i] = make([]float64, T)
	}

	//Init
	for i := 0; i < h.N; i++ {
		a[i][0] = (-1.0 * math.Log(h.Pi[i])) - math.Log(h.B[i][observation[0]])
	}
	//Induction
	for t := range observation {
		for j := 0; j < h.N; j++ {
			min_state = 0
			min_weight = a[0][t - 1] - math.Log(h.A[0][j])

			for i := 0; i < h.N; i++ {
				weigth := a[i][t - 1] - math.Log(h.A[i][j])

				if weigth < min_weight {
					min_state, min_weight = i, weigth
				}
			}
			a[j][t] = min_weight - math.Log(h.B[j][observation[t]])
			s[j][t] = min_state
		}
	}

	// Min for the last element of observation
	min_state = 0
	min_weight = a[0][T - 1]

	for i := 1; i < h.N; i++ {
		if a[i][T - 1] < min_weight {
			min_state = i
			min_weight = a[i][T - 1]
		}
	}

	//Traceback
	path := make([]int, T)
	path[T - 1] = min_state

	for t := T - 2; t >= 0; t-- {
		path[t] = s[path[t + 1]][t + 1]
	}

	*probability = math.Exp(-min_weight)
	return path
}


func (h *HiddenMM) Store() {
	pool := dbconn.Pool
	connection := pool.Get()
	
	fmt.Println("Saving transition matrix A...")
	connection.Send("MULTI")
	for i := range h.A {
		connection.Send("LPUSH", DBA + string(i), h.A[i])
	}
	connection.Do("EXEC")

	fmt.Println("Saving emission matrix B...")
	connection.Send("MULTI")
	for i := range h.B {
		connection.Send("LPUSH", DBA + string(i), h.B[i])
	}
	connection.Do("EXEC")

	fmt.Println("Saving probability distribution Pi...")
	connection.Do("LPUSH", DBPI, h.Pi)

	connection.Close()
}


func Load() HiddenMM {
	pool := dbconn.Pool
	connection := pool.Get()

	A := make([][]float64, NSTATES, NSTATES)

	for i := range A {
		A[i] = make([]float64, NSTATES, NSTATES)
		A[i], err := connection.Do("LRANGE", DBA + string(i), 1, -1)

		if err != nil {
			fmt.Println("Error while reading A[", i, "]")
			os.Exit(1)
		}
	}

	B := make([][]float64, hmm.NSTATES, NSTATES)

	for i := range B {
		B[i] = make([]float64, NVALS, NVALS)
		B[i], err = connection.Do("LRANGE", DBB + string(i), 1, -1)

		if err != nil {
			fmt.Println("Error while reading B[", i, "]")
			os.Exit(1)
		}
	}

	Pi := make([]float64, NSTATES, NSTATES)
	Pi, err = connection.Do("LRANGE", DBPI, 1, -1)

	if err != nil {
		fmt.Println("Error while reading Pi")
		os.Exit(1)
	}
	connection.Close()

	return HiddenMM{NSTATES, NVALS, A, B, Pi}
}


// func main() {
// 	dir, err := os.Open(FULL)

// 	if err != nil {
// 		fmt.Println("Error reading directory", FULL)
// 		os.Exit(1)
// 	}

// 	// select all filenames from open directory
// 	files_slice, err := dir.Readdirnames(0)

// 	if err != nil {
// 		fmt.Println("Error reading filenames from directory", FULL)
// 		os.Exit(1)
// 	}	

// 	err = dir.Close()

// 	if err != nil {
// 		fmt.Println("Error closing directory", FULL)
// 		os.Exit(1)
// 	}
// 	hmm := InitHMM(NSTATES, NVALS)

// 	observed_sequence := make([][]int, len(files_slice))

// 	for i, f := range files_slice {
// 		fmt.Println("Processing", f, "file")
// 		// learn only sequences for now
// 		_, observed_sequence[i] = ObserveFile(f)
// 	}

// 	fmt.Println("Begin learning...")
// 	hmm.Learn(observed_sequence, 0, 0.01)
	
// 	fmt.Println("Saving model in database...")
// 	hmm.Store()

// 	fmt.Println("Learning succeed!")
// }