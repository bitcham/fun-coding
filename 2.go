package main

import (
	"fmt"
	"strings"
)

func FormatTable(headers []string, rows [][]string) string {
	// TODO: Create aligned ASCII table
	// 1. Calculate max width for each column
	// 2. Print header row with proper widths
	// 3. Print separator line of dashes
	// 4. Print each data row
	// Use fmt.Sprintf("%-*s", width, value) for left-aligned formatting
	_ = fmt.Sprintf
	_ = strings.Repeat
	return ""
}

func main() {
	headers := []string{"Name", "Age", "City"}
	rows := [][]string{
		{"Alice", "30", "New York"},
		{"Bob", "25", "LA"},
		{"Charlie", "35", "Chicago"},
	}

	fmt.Print(FormatTable(headers, rows))
}
