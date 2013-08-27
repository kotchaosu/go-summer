package main

import (
	"bufio"
	"fmt"
	"os"
	"nlptk"
	"sort"
	"strings"
)

type Rater struct {
	number int
	rate float64
}

type By func(r1, r2 *Rater) bool

func (by By) Sort(r []Rater) {
	rs := &RateSorter{
		rates: r,
		by: by,
	}
	sort.Sort(rs)
}

type RateSorter struct {
	rates []Rater
	by func(r1, r2 *Rater) bool
}

func (r *RateSorter) Len() int {
	return len(r.rates)
}

func (r *RateSorter) Swap(i, j int) {
	r.rates[i], r.rates[j] = r.rates[j], r.rates[i]
}

func (r *RateSorter) Less(i, j int) bool {
	return r.by(&r.rates[i], &r.rates[j])
}


func main() {
	arguments := os.Args

	if len(arguments) < 2 {
		fmt.Println("Needed argument")
		os.Exit(1)
	}

	filename := arguments[1]
	file, err := os.Open(filename)

	if err != nil {
		fmt.Println("Error reading file", filename)
		os.Exit(1)
	}

	reader := bufio.NewReader(file)

	p_counter := 0
	s_counter := 0
	is_first := true

	first_sentence := nlptk.Sentence{}
	text_structure := make([]nlptk.Paragraph, 10, 10)
	// word_counter := make(map[string]int)
	rates := make([]Rater, 10, 10)
	sentence_list := make([]nlptk.Sentence, 10, 10)

	for bpar, e := reader.ReadBytes('\n'); e == nil; bpar, e = reader.ReadBytes('\n') {

		if len(string(bpar)) == 0 {
			continue
		}

		paragraph := nlptk.Paragraph{p_counter, string(bpar)}
		text_structure[p_counter] = paragraph

		if is_first {
			is_first = false
			first_sentence = nlptk.Sentence{0, paragraph.GetParts()[0]}
		}
		first_bigrams := first_sentence.CreateBigrams()

		sentences := paragraph.GetParts()

		if len(sentences) != 0 {			
			for _, sentence := range sentences {

				coverage := 0.0
				s := nlptk.Sentence{s_counter, sentence}
				words := s.GetParts()
				bigrams := s.CreateBigrams()

				if len(words) * len(bigrams) == 0 {
					continue
				}

				for _, bigram := range bigrams {
					if first_sentence.IsIn(bigram[0]) {
						coverage += 1.0
						if first_sentence.IsIn(bigram[1]) {
							coverage += 0.75
						}
					}
					for _, v := range first_bigrams {
						if bigram[0] == v[0] && bigram[1] == v[1] {
							coverage += 0.75
						}
					}
				}

				rates = append(rates, Rater{s_counter, float64(coverage)/float64(len(words))})
				sentence_list = append(sentence_list, s)

				s_counter++

				if s_counter == len(rates) {
					temp := make([]Rater, len(rates)+5, len(rates)+5)
					copy(temp, rates)
					rates = temp
				}
				
				if s_counter == len(sentence_list) {
					temp := make([]nlptk.Sentence, len(sentence_list)+5, len(sentence_list)+5)
					copy(temp, sentence_list)
					sentence_list = temp
				}

			}	
			p_counter++

			if p_counter == len(text_structure) {
				temp := make([]nlptk.Paragraph, len(text_structure)+5, len(text_structure)+5)
				copy(temp, text_structure)
				text_structure = temp
			}
		}
	}

	rate := func(r1, r2 *Rater) bool {
		return r1.rate > r2.rate
	}

	number := func(r1, r2 *Rater) bool {
		return r1.number < r2.number
	}

	By(rate).Sort(rates)

	rates = rates[:4]
	
	By(number).Sort(rates)

	for i := range(rates) {
		out_sentence := sentence_list[rates[i].number].Text
		out_sentence = strings.SplitAfter(out_sentence, "\"")[0]
		fmt.Println(out_sentence)
	}
}
