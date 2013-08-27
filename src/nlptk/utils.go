package nlptk

import "strings"

type Texter interface {
	// returns slice of smaller parts inside
	// for paragraphs -> sentences
	// for sentence -> words
	GetParts() []string
	// checks if text attribute contains string
	IsIn(str string) bool 
	CreateBigrams () [][2]string
}

type Paragraph struct {
	Number int
	Text string
}

func (p *Paragraph) GetParts () []string {
	str := p.Text
	return strings.Split(str, ".")
}

func (p *Paragraph) IsIn (str string) bool {
	if strings.Count(p.Text, str) != 0 {
		return true
	}
	return false
}

type Sentence struct {
	Number int
	Text string
}

func (s *Sentence) GetParts () []string {
	str := s.Text
	return strings.Fields(str)
}

func (s *Sentence) IsIn (str string) bool {
	if strings.Count(s.Text, str) != 0 {
		return true
	}
	return false
}

func (s *Sentence) CreateBigrams () [][]string {
	str := s.Text
	unigrams := strings.Fields(str)
	
	if len(unigrams) == 0 {
	   return nil
	}

	for i, v := range(unigrams) {
	    unigrams[i] = strings.Trim(v, "!?,.'()")
	}

	bigrams := make([][]string, len(unigrams)+1, len(unigrams)+1)

	for i, v := range unigrams {
		switch i {
		case 0:
			bigrams[i] = []string{"", v}
		default:
			bigrams[i] = []string{unigrams[i-1], v}
		}
	}
	bigrams[len(unigrams)] = []string{unigrams[len(unigrams)-1], ""}

	return bigrams
}

// dictionary for preparing bigram statistics
// maps words to its successors in bigrams
// and next to number of times the successor appeared
type Dict map[string]map[string]int

func (d Dict) CreateDictionary(s *Sentence) {
	b := s.CreateBigrams()
	for _, v := range b {
		if mm, ok := d[v[0]]; !ok {
			mm = make(map[string]int)
			d[v[0]] = mm
		}
		d[v[0]][v[1]]++
	}
}

func (d Dict) showProbability (str1, str2 string) float64 {
     den := 0
     for _, v := range (d[str1]) {
     	 den += v
     }
     return float64(d[str1][str2])/float64(den)
}
