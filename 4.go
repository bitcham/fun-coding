package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type Book struct {
	ID    int    `json:"id"`
	Title string `json:"title"`
}

func NewAuthenticatedRequest(method, url, token string) (*http.Request, error) {
	// TODO: Create request with http.NewRequest
	// TODO: Set Authorization header to "Bearer " + token
	req, err := http.NewRequest(method, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	return req, nil
}

func FetchBookWithAuth(id int, token string) (string, error) {
	// TODO: Build URL for http://bookstore-api/books/{id}
	// TODO: Use NewAuthenticatedRequest to create the request
	// TODO: Execute with client.Do(req)
	// TODO: Check status, parse JSON, return title
	url := fmt.Sprintf("http://bookstore-api/books/%d", id)
	req, err := NewAuthenticatedRequest("GET", url, token)
	if err != nil {
		return "", err
	}
	client := &http.Client{
		Timeout: 10 * time.Second,
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to fetch book: %d", resp.StatusCode)
	}

	var book Book
	if err := json.NewDecoder(resp.Body).Decode(&book); err != nil {
		return "", err
	}

	return book.Title, nil
}

func main() {
	title, err := FetchBookWithAuth(1, "my-token")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println(title)
}
