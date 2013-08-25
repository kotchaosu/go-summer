package nlptk

import "strings"

type Texter interface {
	// order of paragraph/sentence/word matters
	GetNumber () int
	// returns number of smaller parts inside
	// for paragraphs -> sentences
	// for sentence -> words
	CountParts() int
	GetText() string
	// checks if text attribute contains string
	IsIn(str string) bool 
	CreateBigrams () [][2]string
}

type Paragraph struct {
	number, numsentences int
	text string
}

func (p *Paragraph) GetNumber () int {
	return p.number
}

func (p *Paragraph) CountParts () int {
	return p.numsentences
}

func (p *Paragraph) GetText () string {
	return p.text
}

func (p *Paragraph) IsIn (str string) bool {
	text := p.GetText()
	for i := 0; i < len(text) - len(str); i++ {
		if str == text[i:i+len(str)-1] {
			return true
		}
	}
	return false
}

type Sentence struct {
	number, numwords int
	text string
}

func (s *Sentence) GetNumber () int {
	return s.number
}

func (s *Sentence) CountParts () int {
	return s.numwords
}

func (s *Sentence) GetText () string {
	return s.text
}

func (s *Sentence) IsIn (str string) bool {
	text := s.GetText()
	for i := 0; i < len(text) - len(str); i++ {
		if str == text[i:i+len(str)-1] {
			return true
		}
	}
	return false
}

func (s *Sentence) CreateBigrams () [][2]string {
	str := s.GetText()
	unigrams := strings.Fields(str)
	
	if len(unigrams) == 0 {
	   return nil
	}

	for i, v := range(unigrams) {
	    unigrams[i] = strings.Trim(v, "!?,.'()")
	}

	bigrams := make([][2]string, len(unigrams)+1, len(unigrams)+1)

	for i, v := range unigrams {
		switch i {
		case 0:
			bigrams[i] = [2]string{"", v}
		default:
			bigrams[i] = [2]string{unigrams[i-1], v}
		}
	}
	bigrams[len(unigrams)] = [2]string{unigrams[len(unigrams)-1], ""}

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
	