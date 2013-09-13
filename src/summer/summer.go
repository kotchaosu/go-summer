// summarization

package main

import (
	"os"
	"bufio"
	"hmm1"
	"fmt"
	"nlptk"
	"strings"
	)

const(
	FULL = "/home/maxabsent/Documents/learning_set/full_texts/"
	SUMM = "/home/maxabsent/Documents/learning_set/summarizations/"
	FILE = "/home/maxabsent/Documents/learning_set/evaluation/text_0"

	HEVAL = "/home/maxabsent/Documents/learning_set/evaluation/human/"
	AEVAL = "/home/maxabsent/Documents/learning_set/evaluation/auto/0/"
	FILENAME = "text_0"
)


func GetFileReader(filename string) *bufio.Reader {
	file, err := os.Open(filename)

	if err != nil {
		fmt.Println("Error reading file", filename)
		os.Exit(1)
	}

	return bufio.NewReader(file)
}

// Version for case #0 position in paragraph
func CreateObservationSequence(filename string, length int) []int {
	output := make([]int, 0, 0)
	
	sentence_number := 0
	reader_full := GetFileReader(filename)

	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
		var sentences []string
		paragraph := nlptk.Paragraph{0, string(bpar)}

		if sentences = paragraph.GetParts(); len(paragraph.Text) <= 1 || len(sentences) == 0 {
			continue
		}

		for i := range sentences {
			output = append(output, i)
		}

		if sentence_number++; 2 * sentence_number + 1 >= length {
			return output
		}
	}
	fmt.Println("Created sequence:", output)
	return output
}

// Version for case #1 sentence length
// func CreateObservationSequence(filename string, length int) []int {
// 	output := make([]int, 0, 0)
	
// 	sentence_number := 0
// 	reader_full := GetFileReader(filename)

// 	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
// 		var sentences []string
// 		paragraph := nlptk.Paragraph{0, string(bpar)}

// 		if sentences = paragraph.GetParts(); len(paragraph.Text) <= 1 || len(sentences) == 0 {
// 			continue
// 		}

// 		for _, s := range sentences {
// 			sentence := nlptk.Sentence{sentence_number, s[:len(s)-1]}
// 			output = append(output, len(sentence.GetParts()))
// 		}

// 		if sentence_number++; 2 * sentence_number + 1 >= length {
// 			return output
// 		}
// 	}
// 	fmt.Println("Created sequence:", output)
// 	return output
// }

// Prints sequence of states (appear, not appear) given by slice 
func PrintSequence(filename string, sequence []int) string {
	output := make([]string, 0, 0)
	sentence_number := 0  // AKA states counter

	reader_full := GetFileReader(filename)

	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
		paragraph := nlptk.Paragraph{0, string(bpar)}

		if len(paragraph.Text) <= 1 {
			continue
		}

		sentences := paragraph.GetParts()

		if len(sentences) == 0 {
			continue
		}

		for _, s := range sentences {
			if sequence[sentence_number] == 1 {
				output = append(output, s)
			}

			sentence_number++
			if 2 * sentence_number + 1 >= len(sequence) {
				return strings.Join(output, ". ")
			}
		}
	}
	return strings.Join(output, ". ")
}
	

// // Main function for summarization
// func Summarize(filename string, h *hmm1.HiddenMM) string {
	
// 	sentence_number := 0  // AKA states counter
// 	paragraph_number := 0
// 	current_state := 0

// 	summarization := make([]string, 0, 0)
// 	output_seq := make([]int, hmm1.NSTATES)

// 	reader_full := GetFileReader(filename)

// 	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
// 		paragraph := nlptk.Paragraph{paragraph_number, string(bpar)}

// 		if len(paragraph.Text) <= 1 {
// 			continue
// 		}

// 		sentences := paragraph.GetParts()

// 		if len(sentences) == 0 {
// 			continue
// 		}

// 		for snum, s := range sentences {
// 			// search highest probability through transition matrix
// 			max_prob := 0.0
// 			next_state := 0

// 			for i := current_state + 1; i < h.N; i++ {
// 				v := h.A[current_state][i]
// 				if v > max_prob {
// 					max_prob = v
// 					next_state = i
// 				}
// 			}

// 			if current_state % 2 != 0 {
// 				summarization = append(summarization, s)
// 				output_seq[2 * sentence_number + 1] = snum + 1
// 			} else {
// 				output_seq[2 * sentence_number] = snum + 1
// 			}

// 			current_state = next_state

// 			sentence_number++
// 			// safety switch
// 			if 2 * sentence_number + 1 == h.N {
// 				return strings.Join(summarization, ". ")
// 			}
// 		}
// 	}
// 	fmt.Println(output_seq)
// 	return strings.Join(summarization, ". ")
// }

func main() {
	// hmm1.Educate(FULL, SUMM, 8, 0.01)
	// read model from db
	markovmodel := hmm1.Load()
	// print summarization
	likelihood := 0.0

	// fmt.Println(Summarize(FILE, &markovmodel))

	input_seq := CreateObservationSequence(FILE, hmm1.NSTATES)
	vitout := markovmodel.Viterbi(input_seq, &likelihood)
	
	fmt.Println(vitout)
	fmt.Println(PrintSequence(FILE, vitout))

	fmt.Println(hmm1.EvaluateSummary(FILENAME, HEVAL, AEVAL))
}
