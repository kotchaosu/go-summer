// summarization

package main

import (
	"os"
	"bufio"
	"hmm"
	"fmt"
	"dbconn"
	"redis/redis"
	)


const FILE = "/home/maxabsent/Documents/"


func ObserveFile(filename string, h *hmm.HiddenMM) ([][]float64, []int) {
	full, err := os.Open(filepath)

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
			for i, v := range h.A[current_state] {
				if v > max_prob {
					max_prob = v
					next_state = i
				}
			}
			current_state = next_state

			if current_state % 2 != 0 {
				summarization = append(summarization, s)
			}

			sentence_number++
			// safety switch
			if sentence_number == h.N {
				return summarization
			}
		}
	}

	return summarization
}

func main() {
	// read model from db
	markovmodel := hmm.Load()
	// open and process file
	for _, v := range Summarize(FILE, &markovmodel) {
		// print summarization
		fmt.Println(v, ".")
	}
}
