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

const(
	FULL = "/home/maxabsent/Documents/learning_set/full_texts/"
	SUMM = "/home/maxabsent/Documents/learning_set/summarizations/"
	FILE = "/home/maxabsent/Documents/learning_set/evaluation/text_0"

	HEVAL = "/home/maxabsent/Documents/learning_set/evaluation/human/"
	AEVAL = "/home/maxabsent/Documents/learning_set/evaluation/auto/0/"
	FILENAME = "text_0"

	FEATURE = 1

	N = 200
	M = 100
)


func GetFileReader(filename string) *bufio.Reader {
	file, err := os.Open(filename)

	if err != nil {
		fmt.Println("Error reading file", filename)
		os.Exit(1)
	}

	return bufio.NewReader(file)
}


// Analyze full text and summarization to prepare observations:
//	- vectors of features
//	- binary table of sentence in summarization presence
func ObserveFile(filename, full_dir, summ_dir string, feature int) []int {

	full, err := os.Open(full_dir + filename)

	if err != nil {
		fmt.Println("Error reading file", full_dir + filename)
		os.Exit(1)
	}

	summ, err := os.Open(summ_dir + filename)

	if err != nil {
		fmt.Println("Error reading file", summ_dir + filename)
		os.Exit(1)
	}
	sentence_counter := make([]int, 0, 0)

	sentence_number, paragraph_number := 0, 0

	reader_summ := bufio.NewReader(summ)
	spar, _ := reader_summ.ReadBytes('\n')
	summarization := nlptk.Paragraph{paragraph_number, string(spar)}
	sum_sentences := summarization.Text

	reader_full := bufio.NewReader(full)
	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {

		var sentences []string
		paragraph := nlptk.Paragraph{paragraph_number, string(bpar)}

		if sentences = paragraph.GetParts(); len(paragraph.Text) <= 1 || len(sentences) == 0 {
			continue
		}

		if feature == 0 {
			for i, s := range sentences {
				if strings.Contains(sum_sentences, s[:len(s)-1]) {
					sentence_counter = append(sentence_counter, i)
				}
				if sentence_number++; sentence_number == N {
					return sentence_counter
				}
			}
		} else {
			for _, s := range sentences {
				sentence := nlptk.Sentence{sentence_number, s[:len(s)-1]}
				if strings.Contains(sum_sentences, s[:len(s)-1]) {
					sentence_counter = append(sentence_counter, len(sentence.GetParts()))
				}
				if sentence_number++; sentence_number == N {
					return sentence_counter
				}
			}
		}

		paragraph_number++
	}
	fmt.Println("sequence", filename, len(sentence_counter), sentence_counter)
	return sentence_counter
}


// Function turns file into slice with feature symbols
// 	- feature == 0 -> position in paragraph
//  - feature == 1 -> sentence length // now for else
func CreateObservationSequence(filename string, length, feature int) []int {
	output := make([]int, 0, 0)
	
	sentence_number := 0
	reader_full := GetFileReader(filename)

	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
		var sentences []string
		paragraph := nlptk.Paragraph{0, string(bpar)}

		if sentences = paragraph.GetParts(); len(paragraph.Text) <= 1 || len(sentences) == 0 {
			continue
		}

		if feature == 0 {
			for i := range sentences {
				output = append(output, i)
			}
		} else {
			for _, s := range sentences {
				sentence := nlptk.Sentence{sentence_number, s[:len(s)-1]}
				output = append(output, len(sentence.GetParts()))
			}
		}

		if sentence_number++; 2 * sentence_number + 1 >= length {
			return output
		}
	}
	fmt.Println("Created sequence:", output)
	return output
}

// Prints sequence of states (appear, not appear) given by slice 
func PrintSequence(filename string, sequence []int) string {
	output := make([]string, 0, 0)
	sentence_number := 0  // AKA states counter

	reader_full := GetFileReader(filename)

	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
		paragraph := nlptk.Paragraph{0, string(bpar)}
		var sentences []string

		if len(paragraph.Text) <= 1 {
			continue
		}

		if sentences = paragraph.GetParts(); len(sentences) == 0 {
			continue
		}

		for _, s := range sentences {
			if sequence[sentence_number] == 0 {
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


// Evaluates summarization - return % of sentences in human-created summarization
// which appeared in program-created one
func EvaluateSummary(filename, human_summ_dir, auto_summ_dir string) (float64, float64) {

	human, err := os.Open(human_summ_dir + filename)

	if err != nil {
		fmt.Println("Error reading file", human_summ_dir + filename)
		os.Exit(1)
	}

	auto, err := os.Open(auto_summ_dir + filename)

	if err != nil {
		fmt.Println("Error reading file", auto_summ_dir + filename)
		os.Exit(1)
	}

	coverage := 0

	reader_human := bufio.NewReader(human)
	
	hpar, _ := reader_human.ReadBytes('\n')
	human_summarization := nlptk.Paragraph{0, string(hpar)}

	reader_auto := bufio.NewReader(auto)
	
	apar, _ := reader_auto.ReadBytes('\n')
	auto_summarization := nlptk.Paragraph{0, string(apar)}


	for _, sentence := range human_summarization.GetParts() {
		if auto_summarization.IsIn(sentence) {
			coverage++
		}
	}

	auto_coverage := float64(coverage) / float64(len(auto_summarization.GetParts()))
	human_coverage := float64(coverage) / float64(len(human_summarization.GetParts()))

	return human_coverage, auto_coverage
}


func Educate(full_dir, summ_dir string, N, M, feature, iterations int, tolerance float64) {
	dir, err := os.Open(full_dir)

	if err != nil {
		fmt.Println("Error reading directory", full_dir)
		os.Exit(1)
	}

	// select all filenames from open directory
	files_slice, err := dir.Readdirnames(0)

	if err != nil {
		fmt.Println("Error reading filenames from directory", full_dir)
		os.Exit(1)
	}	

	err = dir.Close()

	if err != nil {
		fmt.Println("Error closing directory", full_dir)
		os.Exit(1)
	}
	hmm := hmm.InitHMM(N, M)

	observed_sequence := make([][]int, len(files_slice))

	for i, f := range files_slice {
		fmt.Println("Processing", f, "file")
		// learn only sequences for now
		observed_sequence[i] = ObserveFile(f, full_dir, summ_dir, feature)
	}

	fmt.Println("Begin learning...")
	hmm.Learn(observed_sequence, iterations, tolerance)
	
	fmt.Println("Saving model in database...")
	hmm.Store()

	fmt.Println("Learning succeed!")
}


func main() {
	Educate(FULL, SUMM, N, M, FEATURE, 8, 0.01)
	// read model from db
	markovmodel := hmm.Load(N, M)
	likelihood := 0.0

	input_seq := CreateObservationSequence(FILE, N, FEATURE)
	vitout := markovmodel.Viterbi(input_seq, &likelihood)
	
	fmt.Println(vitout)
	fmt.Println(PrintSequence(FILE, vitout))

	fmt.Println(EvaluateSummary(FILENAME, HEVAL, AEVAL))
}
