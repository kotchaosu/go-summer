// summarization

package main

import (
	"os"
	"bufio"
	"hmm0"
	"fmt"
	"nlptk"
	"strings"
	)

const(
	FULL = "/home/maxabsent/Documents/learning_set/full_texts/"
	SUMM = "/home/maxabsent/Documents/learning_set/summarizations/"
	FILE = FULL + "text_6"
)
	

func Summarize(filename string, h *hmm0.HiddenMM) string {
	full, err := os.Open(filename)

	if err != nil {
		fmt.Println("Error reading file", filename)
		os.Exit(1)
	}

	summarization := make([]string, 0, 0)
	sentence_number := 0  // AKA states counter
	paragraph_number := 0
	
	current_state := 0

	output_seq := make([]int, 0, 0)

	reader_full := bufio.NewReader(full)
	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
		paragraph := nlptk.Paragraph{paragraph_number, string(bpar)}

		if len(paragraph.Text) <= 1 {
			continue
		}

		sentences := paragraph.GetParts()

		if len(sentences) == 0 {
			continue
		}

		for _, s := range sentences {
			// search highest probability through transition matrix
			max_prob := 0.0
			next_state := 0

			for i := current_state + 1; i < h.N; i++ {
				v := h.A[current_state][i]
				if v > max_prob {
					max_prob = v
					next_state = i
				}
			}

			if current_state % 2 != 0 {
				summarization = append(summarization, s)
				output_seq = append(output_seq, 0)
			}

			current_state = next_state

			sentence_number++
			// safety switch
			if sentence_number == h.N / 2 {
				return strings.Join(summarization, ".")
			}
		}
	}
	fmt.Println(output_seq)
	return strings.Join(summarization, ".")
}

func main() {
	hmm0.Educate(FULL, SUMM)
	// read model from db
	markovmodel := hmm0.Load()
	// print summarization
	likelihood := 0.0
	input_seq := make([]int, 30)

	fmt.Println(Summarize(FILE, &markovmodel))
	fmt.Println(input_seq)
	fmt.Println(markovmodel.Viterbi(input_seq, &likelihood))
}
