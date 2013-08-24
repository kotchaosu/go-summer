package main
import (
  "io/ioutil"
  "os"
  "fmt"
)

func main() {
  arguments := os.Args
  if len(arguments) < 2 {
    fmt.Println("Podaj nazwę pliku, który chcesz wczytać")
    os.Exit(1)
  }

  filename := arguments[1]
  fcontents, err := ioutil.ReadFile(filename)
  if err != nil {
    fmt.Printf("Błąd wczytywania pliku %v \n", filename)
    os.Exit(1)
  }

  os.Stdout.Write(fcontents)
  os.Stdout.Write([]byte("\n"))
}


func ReadLines(filename string) (lines []string, err os.Error) {
  file, err := os.Open(filename)
  if err != nil {
    return
  }

  reader := bufio.NewReader(file)
  for bline, e := reader.ReadBytes('\n'); e == nil; bline, e = reader.ReadBytes('\n') {
    lines = append(lines, string(bline))
  }
  return
}

