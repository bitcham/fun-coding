package main

import (
	"encoding/json"
	"fmt"
	"net/http"
)

type Book struct {
	Title  string  `json:"title"`
	Author string  `json:"author"`
	Price  float64 `json:"price"`
}

// BookHandler returns a book as JSON (status 200)
func BookHandler(w http.ResponseWriter, r *http.Request) {
	book := Book{Title: "The Go Programming Language", Author: "Donovan & Kernighan", Price: 39.99}
	// Set Content-Type to "application/json"
	// Encode book as JSON
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(book)
}

// ProcessingHandler returns 202 Accepted with {"status": "processing"}
func ProcessingHandler(w http.ResponseWriter, r *http.Request) {
	// Set Content-Type to "application/json"
	// Set status code to 202 (http.StatusAccepted)
	// Write JSON: {"status": "processing"}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(map[string]string{"status": "processing"})
}

// ErrorHandler returns 500 with {"error": "database connection failed"}
func ErrorHandler(w http.ResponseWriter, r *http.Request) {
	// Set Content-Type to "application/json"
	// Set status code to 500 (http.StatusInternalServerError)
	// Write JSON: {"error": "database connection failed"}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	json.NewEncoder(w).Encode(map[string]string{"error": "database connection failed"})
}

func main() {
	http.HandleFunc("/book", BookHandler)
	http.HandleFunc("/process", ProcessingHandler)
	http.HandleFunc("/error", ErrorHandler)
	fmt.Println("Handlers registered!")
}
