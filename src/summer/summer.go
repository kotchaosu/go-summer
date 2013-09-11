// summarization

package main

import (
	"os"
	"bufio"
	"hmm"
	"fmt"
	"nlptk"
	"strings"
	)


const FILE = "/home/maxabsent/Documents/learning_set/evaluation/text_0"


func Summarize(filename string, h *hmm.HiddenMM) string {
	full, err := os.Open(filename)

	if err != nil {
		fmt.Println("Error reading file", filename)
		os.Exit(1)
	}

	summarization := make([]string, 0, 0)
	sentence_number := 0  // AKA states counter
	paragraph_number := 0
	
	current_state := 0

	reader_full := bufio.NewReader(full)
	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
		paragraph := nlptk.Paragraph{paragraph_number, string(bpar)}
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
			current_state = next_state

			if current_state % 2 != 0 {
				fmt.Println(s)
				summarization = append(summarization, s)
			}

			sentence_number++
			// safety switch
			if sentence_number == h.N / 2 {
				return strings.Join(summarization, ".")
			}
		}
	}
	return strings.Join(summarization, ".")
}

func main() {
	// hmm.Educate()
	// read model from db
	markovmodel := hmm.Load()
	// print summarization
	fmt.Println(Summarize(FILE, &markovmodel))
}
