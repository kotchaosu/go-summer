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
	// >>> TODO: all of these should be parameters main()
	FULL = "/home/maxabsent/Documents/learning_set/full_texts/"
	SUMM = "/home/maxabsent/Documents/learning_set/summarizations/"
	FILE = "/home/maxabsent/Documents/learning_set/evaluation/text_0"

	HEVAL = "/home/maxabsent/Documents/learning_set/evaluation/human/"
	AEVAL = "/home/maxabsent/Documents/learning_set/evaluation/auto/1/"
	FILENAME = "text_0"

	FEATURE = 1

	N = 200
	M = 100
	// <<<
)


type Cataloger interface {
	GetReader(filename string) *bufio.Reader
	GetWriter(filename string) *bufio.Writer
	OpenDir(dirname string) *os.File
	CloseDir(dir *os.File)
}


func GetReader(filename string) *bufio.Reader {
	file, err := os.Open(filename)

	if err != nil {
		fmt.Println("Error reading file", filename)
		os.Exit(1)
	}
	return bufio.NewReader(file)
}


func GetWriter(filename string) *bufio.Writer {
	file, err := os.Create(filename + "_SUMM")

	if err != nil { panic (err) }

	defer func() {
		if err := file.Close(); err != nil {
            panic(err)
        }
	}()
	// 	err = file.Close()
	// 	if err != nil {
	// 		fmt.Println("Fuck up, chief", filename)
	// 	} else {
	// 		fmt.Println("Error creating file", filename)
	// 	}
	// 	os.Exit(1)
	// }
	return bufio.NewWriter(file)
}


func OpenDir(dirname string) *os.File {
	dir, err := os.Open(dirname)

	if err != nil {
		fmt.Println("Error reading directory", dirname)
		os.Exit(1)
	}
	return dir
}


func CloseDir(dir *os.File) {
	err := dir.Close()

	if err != nil {
		fmt.Println("Error closing directory")
		os.Exit(1)
	}
}


// Analyze full text and summarization to prepare observations:
//	- vectors of features
//	- binary table of sentence in summarization presence
func ObserveFile(filename, full_dir, summ_dir string, states, feature int) []int {

	reader_full := GetReader(full_dir + filename)
	reader_summ := GetReader(summ_dir + filename)

	sentence_counter := make([]int, states, states)
	sentence_number, paragraph_number := 0, 0

	spar, _ := reader_summ.ReadBytes('\n')
	summarization := nlptk.Paragraph{paragraph_number, string(spar)}
	sum_sentences := summarization.Text

	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {

		var sentences []string
		paragraph := nlptk.Paragraph{paragraph_number, string(bpar)}

		if sentences = paragraph.GetParts(); len(paragraph.Text) <= 1 || len(sentences) == 0 {
			continue
		}

		if feature == 0 {
			for i, s := range sentences {
				if strings.Contains(sum_sentences, s[:len(s)-1]) {
					sentence_counter[2 * sentence_number + 1] = i + 1
					// sentence_counter[2 * sentence_number] = 0
				} else {
					// sentence_counter[2 * sentence_number + 1] = 0
					// sentence_counter[2 * sentence_number] = i + 1
					sentence_counter[2 * sentence_number] = 0

				}
				if sentence_number++; 2 * sentence_number >= N {
					return sentence_counter
				}
			}
		} else {
			for _, s := range sentences {
				sentence := nlptk.Sentence{sentence_number, s[:len(s)-1]}
				if strings.Contains(sum_sentences, s[:len(s)-1]) {
					sentence_counter[2 * sentence_number + 1] = len(sentence.GetParts())
					// sentence_counter[2 * sentence_number] = 0
				} else {
					// sentence_counter[2 * sentence_number + 1] = 0
					// sentence_counter[2 * sentence_number] = len(sentence.GetParts())
					sentence_counter[2 * sentence_number] = 0
				}
				if sentence_number++; 2 * sentence_number >= N {
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
func CreateObservationSequence(filename string, states, feature int) []int {
	
	output := make([]int, 0, 0)
	sentence_number := 0

	reader_full := GetReader(filename)

	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
		
		var sentences []string
		paragraph := nlptk.Paragraph{0, string(bpar)}

		if sentences = paragraph.GetParts(); len(paragraph.Text) <= 1 || len(sentences) == 0 {
			continue
		}

		if feature == 0 {
			for i := range sentences {
				output = append(output, 0)
				output = append(output, i + 1)
			}
		} else {
			for _, s := range sentences {
				sentence := nlptk.Sentence{sentence_number, s[:len(s)-1]}
				output = append(output, 0)
				output = append(output, len(sentence.GetParts()))
			}
		}

		if sentence_number++; 2 * sentence_number >= states {
			break
		}
	}
	fmt.Println("Created sequence:", output)
	return output
}

// 	output := make([]int, states, states)
// 	sentence_number := 0

// 	reader_full := GetReader(filename)

// 	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
		
// 		var sentences []string
// 		paragraph := nlptk.Paragraph{0, string(bpar)}

// 		if sentences = paragraph.GetParts(); len(paragraph.Text) <= 1 || len(sentences) == 0 {
// 			continue
// 		}

// 		if feature == 0 {
// 			for i := range sentences {
// 				output[2 * sentence_number] = 0
// 				output[2 * sentence_number + 1] = i + 1
// 			}
// 		} else {
// 			for _, s := range sentences {
// 				sentence := nlptk.Sentence{sentence_number, s[:len(s)-1]}
// 				output[2 * sentence_number] = 0
// 				output[2 * sentence_number + 1] = len(sentence.GetParts())
// 			}
// 		}

// 		if sentence_number++; 2 * sentence_number >= states {
// 			break
// 		}
// 	}
// 	fmt.Println("Created sequence:", output)
// 	return output
// }

// Prints sequence of states (appear, not appear) given by slice 
func PrintSequence(filename string, sequence []int) string {
	
	output := make([]string, 0, 0)
	sentence_number := 0  // AKA states counter

	reader_full := GetReader(filename)
	writer := GetWriter(filename)

    buf := make([]byte, 1024)

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
			if sequence[sentence_number] == 1 {
				output = append(output, s)
				for i := 0; i < len(s); i++ {
					buf[i] = byte(s[i])
				}
				writer.Write(buf)
			}

			if sentence_number++; 2 * sentence_number + 1 >= len(sequence) {
				return strings.Join(output, ". ")
			}
		}
	}
	writer.Flush()
	return strings.Join(output, ". ")
}


// Evaluates summarization - return % of sentences in human-created summarization
// which appeared in program-created one
func EvaluateSummary(filename, human_summ_dir, auto_summ_dir string) (float64, float64) {

	reader_human := GetReader(human_summ_dir + filename)
	reader_auto := GetReader(auto_summ_dir + filename)

	hpar, _ := reader_human.ReadBytes('\n')
	human_summarization := nlptk.Paragraph{0, string(hpar)}
	
	apar, _ := reader_auto.ReadBytes('\n')
	auto_summarization := nlptk.Paragraph{0, string(apar)}

	match_auto := 0
	all_auto := 0

	match_human := 0
	all_human := 0

	for _, sentence := range auto_summarization.GetParts() {
		s := nlptk.Sentence{0, sentence}
		for _, bigram := range s.CreateBigrams() {
			if human_summarization.IsIn(strings.Join(bigram, " ")) {
				match_auto++
			}
			all_auto++
		}
	}

	for _, sentence := range human_summarization.GetParts() {
		s := nlptk.Sentence{0, sentence}
		for _, bigram := range s.CreateBigrams() {
			if auto_summarization.IsIn(strings.Join(bigram, " ")) {
				match_human++
			}
			all_human++
		}
	}

	auto_coverage := float64(match_auto) / float64(all_auto)
	human_coverage := float64(match_human) / float64(all_human)

	return auto_coverage, human_coverage
}


func Educate(full_dir, summ_dir string, N, M, feature, iterations int, tolerance float64) {
	dir := OpenDir(full_dir)

	// select all filenames from open directory
	files_slice, err := dir.Readdirnames(0)

	if err != nil {
		fmt.Println("Error reading filenames from directory", full_dir)
		os.Exit(1)
	}	

	CloseDir(dir)

	hmm := hmm.InitHMM(N, M)

	observed_sequence := make([][]int, len(files_slice))

	for i, f := range files_slice {
		fmt.Println("Processing", f, "file")
		// learn only sequences for now
		observed_sequence[i] = ObserveFile(f, full_dir, summ_dir, N, feature)
	}

	fmt.Println("Begin learning...")
	hmm.Learn(observed_sequence, iterations, tolerance)
	
	fmt.Println("Saving model in database...")
	hmm.Store()

	fmt.Println("Learning succeed!")
}


func main() {
	// Educate(FULL, SUMM, N, M, FEATURE, 8, 0.01)
	// read model from db
	markovmodel := hmm.Load(N, M)
	likelihood := 0.0

	input_seq := CreateObservationSequence(FILE, N, FEATURE)
	vitout := markovmodel.Viterbi(input_seq, &likelihood)
	
	fmt.Println(vitout)
	fmt.Println(PrintSequence(FILE, vitout))

	fmt.Println(EvaluateSummary(FILENAME, HEVAL, AEVAL))
}
