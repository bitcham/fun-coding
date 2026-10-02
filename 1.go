package main

import (
	"fmt"
	"strconv"
	"strings"
)

func ParseConfig(lines []string) map[string]interface{} {
	config := make(map[string]interface{})
	for _, line := range lines {
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		parts[0] = strings.TrimSpace(parts[0])
		parts[1] = strings.TrimSpace(parts[1])
		switch parts[1] {
		case "true":
			config[parts[0]] = true
		case "false":
			config[parts[0]] = false
		default:
			if i, err := strconv.Atoi(parts[1]); err == nil {
				config[parts[0]] = i
			} else if f, err := strconv.ParseFloat(parts[1], 64); err == nil {
				config[parts[0]] = f
			} else {
				config[parts[0]] = parts[1]
			}
		}
	}
	return config
}

func main() {
	lines := []string{
		"port=8080",
		"debug=true",
		"rate=1.5",
		"name=MyApp",
		"count = 42",
		"enabled = false",
		"invalid line",
		"pi=3.14159",
	}

	config := ParseConfig(lines)

	// Print in sorted order for consistent output
	keys := []string{"port", "debug", "rate", "name", "count", "enabled", "pi"}
	for _, k := range keys {
		if v, ok := config[k]; ok {
			fmt.Printf("%s = %v (%T)\n", k, v, v)
		}
	}
}
