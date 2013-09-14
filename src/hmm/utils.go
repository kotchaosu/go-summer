package hmm

// import (
// 	"nlptk"
// 	"math"
// 	"dict"
// 	)

// // Builds vectors of features for every sentence
// //
// // Feature vector:
// //	1. position within paragraph
// //	2. number of terms in sentence
// //	3. how likely the terms are given the baseline of terms
// //	4. how likely the terms are given the document terms
// func ComputeFeatures(sentence nlptk.Sentence, document_dict map[string]int) []float64 {
	
// 	all_words := float64(dict.GetWordCount("TOTAL"))
// 	all_doc_words := float64(document_dict["TOTAL"])	

// 	words := sentence.GetParts()

// 	number_of_sentence := float64(sentence.Number)
// 	number_of_terms := float64(len(words))
// 	baseline_likelihood := 0.0
// 	document_likelihood := 0.0

// 	for _, word := range words {
// 		base_count := float64(dict.GetWordCount(word))
// 		doc_count := float64(document_dict[word])

// 		baseline_likelihood += math.Log10(base_count/all_words)
// 		document_likelihood += math.Log10(doc_count/all_doc_words)
// 	}

// 	features := [...]float64{
// 		float64(number_of_sentence),
// 		float64(number_of_terms),
// 		baseline_likelihood,
// 		document_likelihood
// 	}
// 	return features
// }