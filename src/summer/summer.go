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

const (
	// model parameters
	N = 100
	M0 = 10
	M1 = 100
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
	file, err := os.Create(filename + "_s")

	defer file.Close()
    if err != nil {
		fmt.Println("Error reading file", filename)
		os.Exit(1)
	}
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
func ObserveFile(filename, full_dir, summ_dir string, states int) [][]int {

	reader_full := GetReader(full_dir + filename)
	reader_summ := GetReader(summ_dir + filename)

	sentence_counter := make([][]int, 0, 0)
	sentence_number, paragraph_number := 0, 0

	spar, _ := reader_summ.ReadBytes('\n')
	summarization := nlptk.Paragraph{paragraph_number, string(spar)}
	sum_sentences := summarization.Text

	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {

		var sentences []string
		paragraph := nlptk.Paragraph{paragraph_number, string(bpar[:len(bpar)-1])}

		if sentences = paragraph.GetParts(); len(paragraph.Text) <= 1 || len(sentences) == 0 {
			continue
		}

		for i, s := range sentences {
			sentence := nlptk.Sentence{sentence_number, s[:len(s)-1]}
			
			if strings.Contains(sum_sentences, s[:len(s)-1]) { 
				// summary
				sentence_counter = append(sentence_counter, []int{0, 0})
				sentence_counter = append(sentence_counter, []int{i + 1, len(sentence.GetParts())})
			} else {
				// non-summary
				sentence_counter = append(sentence_counter, []int{i + 1, len(sentence.GetParts())})
				sentence_counter = append(sentence_counter, []int{0, 0})
			}
			
			if sentence_number++; 2 * sentence_number >= states {
				return sentence_counter
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
func CreateObservationSequence(filename string, states int) [][]int {
	
	output := make([][]int, 0, 0)

	sentence_number := 0

	reader_full := GetReader(filename)

	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {
		
		var sentences []string
		paragraph := nlptk.Paragraph{0, string(bpar[:len(bpar)-1])}

		if sentences = paragraph.GetParts(); len(paragraph.Text) <= 1 || len(sentences) == 0 {
			continue
		}

		for i, s := range sentences {			
			sentence := nlptk.Sentence{sentence_number, s[:len(s)-1]}

			output = append(output, []int{i + 1, len(sentence.GetParts())}) // summary state
		}
		if sentence_number++; 2 * sentence_number >= states {
			break
		}
	}

	fmt.Println("Created sequence:", output)
	return output
}


// Prints sequence of states (appear, not appear) given by slice 
func PrintSequence(filename string, sequence []int) string {
	
	output := make([]string, 0, 0)
	sentence_number := 0  // AKA states counter

	reader_full := GetReader(filename)
	writer := GetWriter(filename)

    // buf := make([]byte, 1024)

	for bpar, e := reader_full.ReadBytes('\n'); e == nil; bpar, e = reader_full.ReadBytes('\n') {

		paragraph := nlptk.Paragraph{0, string(bpar[:len(bpar)-1])}
		var sentences []string

		if len(paragraph.Text) <= 1 {
			continue
		}

		if sentences = paragraph.GetParts(); len(sentences) == 0 {
			continue
		}

		for _, s := range sentences {
			for _, v := range sequence {
				if v == 2 * sentence_number + 1 {
					output = append(output, s)
					writer.Write([]byte(s))
				}
			}

			if sentence_number++; 2 * sentence_number + 1 >= len(sequence) {
				fmt.Println(writer.Flush())
				return strings.Join(output, ". ")
			}
		}
	}
	writer.Flush()
	return strings.Join(output, ". ")
}


func Educate(full_dir, summ_dir string, N, M0, M1 int) {
	dir := OpenDir(full_dir)

	// select all filenames from open directory
	files_slice, err := dir.Readdirnames(0)

	if err != nil {
		fmt.Println("Error reading filenames from directory", full_dir)
		os.Exit(1)
	}	

	CloseDir(dir)

	hmm := hmm.InitHMM(N, M0, M1)

	observed_sequence := make([][][]int, len(files_slice))

	for i, f := range files_slice {
		fmt.Println("Processing", f, "file")
		// learn only sequences for now
		observed_sequence[i] = ObserveFile(f, full_dir, summ_dir, N)
	}

	fmt.Println("Begin learning...")
	hmm.Learn(observed_sequence)
	
	fmt.Println("Saving model in database...")
	hmm.Store()

	fmt.Println("Learning succeed!")
}

func main() {
	arguments := os.Args

	var filename, full, summ string

    switch len(arguments) {
    case 2:
    	filename = arguments[1]
		full = "/home/maxabsent/Documents/learning_set/full_texts/"
		summ = "/home/maxabsent/Documents/learning_set/summarizations/"
    case 4:
    	filename = arguments[1]
		full = arguments[2]
		summ = arguments[3]
	default:
      	fmt.Println(">>> Go Summer usage <<<")
      	fmt.Println("BASIC USAGE")
      	fmt.Println("1st arg -> path with file to summarize")
      	fmt.Println("LEARNING")
      	fmt.Println("2nd arg -> path with full texts")
      	fmt.Println("3rd arg -> path with summaries")
        os.Exit(1)
    }


	Educate(full, summ, N, M0, M1)
	// read model from db
	markovmodel := hmm.Load(N, M0, M1)
	likelihood := 0.0

	input_seq := CreateObservationSequence(filename, N)
	vitout := markovmodel.Viterbi(input_seq, &likelihood)

	fmt.Println(vitout)
	fmt.Println("", PrintSequence(filename, vitout))
	fmt.Println("Result sequence probability", markovmodel.Evaluate(input_seq, false))
}