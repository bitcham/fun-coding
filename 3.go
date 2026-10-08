package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Book represents the API response (only fields we need)
type Book struct {
	ID    int    `json:"id"`
	Title string `json:"title"`
}

func GetBookTitle(id int) (string, error) {
	// TODO: Implement this function
	// 1. Build the URL: http://bookstore-api/books/{id}
	// 2. Make a GET request using http.Get()
	// 3. Don't forget to close the response body!
	// 4. Check if status code is http.StatusOK
	// 5. Parse the JSON response into a Book struct
	// 6. Return the title
	url := fmt.Sprintf("http://bookstore-api/books/%d", id)
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP status code %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	var book Book
	err = json.Unmarshal(body, &book)
	if err != nil {
		return "", err
	}

	return book.Title, nil
}

func main() {
	title, _ := GetBookTitle(1)
	fmt.Println(title)
}
