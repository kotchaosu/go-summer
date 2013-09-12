// package extracts and counts all words from learning set
package dict

import (
	"bufio"
	"fmt"
	"nlptk"
	"os"
	"dbconn"
	"redis/redis"
)

const (
	DBDICTPREFIX = "#"
)


func GetWordCount(word string) int {
	pool := dbconn.Pool
	connection := pool.Get()

	reply, err := redis.Values(connection.Do("GET", DBDICTPREFIX + word))
	
	if err != nil {
		fmt.Println("DB: Error reading word", word)
		os.Exit(1)
	}
	connection.Close()

	count, _ := redis.Int(reply, err)
	return count	
}

// Extracts, trims from special signs and counts "bare" words in learning set.
// Result dictionary is sent to channel.
func WordCount(filename string, dict chan map[string]int) {
	word_counter := make(map[string]int)
	file, err := os.Open(SETDIR + filename)

	if err != nil {
		fmt.Println("Error reading file", filename)
		os.Exit(1)
	}

	reader := bufio.NewReader(file)

	for bpar, e := reader.ReadBytes('\n'); e == nil; bpar, e = reader.ReadBytes('\n') {
		paragraph := nlptk.Paragraph{0, string(bpar)}
		sentences := paragraph.GetParts()

		for _, sentence := range sentences {
			s := nlptk.Sentence{0, sentence}
			words := s.GetParts()

			if len(words) == 0 {
				continue
			}

			for _, word := range words {
				word_counter[word]++
				word_counter["TOTAL"]++
			}
		}
	}
	dict <- word_counter
}

func Store(dictionary chan map[string]int) {
	pool := dbconn.Pool
	connection := pool.Get()

	for d := range dictionary {
		for k, v := range d {

			k = DBDICTPREFIX + k

			_, err := connection.Do("EXISTS", k)
			if err != nil {
				connection.Do("SET", k, v) // create new entry
			} else {
				connection.Do("INCRBY", k, v) // update existing
			}
		}
	}
	connection.Close()
}

func Learn(dirpath string) {
	dir, err := os.Open(dirpath)

	if err != nil {
		fmt.Println("Error reading directory", dirpath)
		os.Exit(1)
	}

	// select all filenames from open directory
	files_slice, err := dir.Readdirnames(0)

	if err != nil {
		fmt.Println("Error reading filenames from directory", dirpath)
		os.Exit(1)
	}

	err = dir.Close()

	if err != nil {
		fmt.Println("Error closing directory", dirpath)
		os.Exit(1)
	}

	dict := make(chan map[string]int)

	go func() {
		for _, f := range files_slice {
			WordCount(f, dict)
		}
		close(dict)
	}()
	Store(dict)

	fmt.Println("Learning succeed!")
}
