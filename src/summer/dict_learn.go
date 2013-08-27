// learning from source files
// updating/retrieving information from knowledge DB

package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"
)

const (
	SETDIR = "~/bin/learning_set/"
)

func main() {
	// select every file from set directory
	dir, err := os.Open(SETDIR)

	if err != nil {
		fmt.Println("Error reading directory", SETDIR)
		os.Exit(1)
	}

	files_slice, err := dir.Readdirnames(0)

	if err != nil {
		fmt.Println("Error reading filenames from directory", SETDIR)
		os.Exit(1)
	}	

	// open and maintain operations -> opportunity for concurrency
	for _, v := range(files_slice) {
		Learn(SETDIR + v)
	}

	err := dir.Close()

	if err != nil {
		fmt.Println("Error closing directory", SETDIR)
		os.Exit(1)
	}

	fmt.Println("Learning succeed!")
}
