package main

import (
	"fmt"
	"net/http"
)

func HelloHandler(w http.ResponseWriter, r *http.Request) {
	// Write "Hello, Gopher!" to the response
	// Use: fmt.Fprint(w, "your message")
	fmt.Fprint(w, "Hello, Gopher!")
}

func main() {
	// Register HelloHandler to handle "/hello"
	// Use: http.HandleFunc("/hello", HelloHandler)
	http.HandleFunc("/hello", HelloHandler)

	fmt.Println("Server ready!")
}
