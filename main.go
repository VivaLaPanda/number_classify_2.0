package main

import (
	"bufio"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"strings"

	neuralNet "github.com/vivalapanda/number_classify_2.0/NeuralNetwork"
	featLib "github.com/vivalapanda/number_classify_2.0/features"
)

func main() {
	trainRawSlice, trainExptValInt := parseFile("train_data.txt")
	testRawSlice, testExptValInt := parseFile("test_data.txt")

	// Extracting those features
	trainSlice := [][]float64{}
	for _, imageSlice := range trainRawSlice {
		trainSlice = append(trainSlice, featLib.RegionAvg(imageSlice))
	}
	testSlice := [][]float64{}
	for _, imageSlice := range testRawSlice {
		testSlice = append(testSlice, featLib.RegionAvg(imageSlice))
	}

	// Neural net expects floats
	trainExptVal := make([][]float64, len(trainExptValInt))
	testExptVal := make([][]float64, len(testExptValInt))
	for i, numeral := range trainExptValInt {
		trainExptVal[i] = make([]float64, 10)
		trainExptVal[i][numeral] = 1
	}
	for i, numeral := range testExptValInt {
		testExptVal[i] = make([]float64, 10)
		testExptVal[i][numeral] = 1
	}

	// Testing different hidden layers
	for i := 10; i < 51; i += 10 {
		nn := neuralNet.NewNeuralNetwork(25, i, 10)

		// Train it
		for epoch := 0; epoch < 500; epoch++ {
			for j, input := range trainSlice {
				nn.Backpropagation(input, trainExptVal[j], .01, 0)
			}
		}

		// Test it
		fmt.Println("  Ans    :    NN")

		successCount := 0
		totalProcessesed := 0

		for j, input := range testSlice {
			outputSlice, err := nn.Calc(input)
			if err != nil {
				fmt.Println(err.Error())
				return
			}

			maxIndex := 0
			maxVal := 0.0
			for k, element := range outputSlice {
				if element > maxVal {
					maxIndex = k
					maxVal = element
				}
			}

			outputSlice = make([]float64, len(outputSlice))
			outputSlice[maxIndex] = 1

			if reflect.DeepEqual(testExptVal[j], outputSlice) {
				successCount++
			}

			totalProcessesed++
		}

		fmt.Printf("Hidden Neurons: %v\n", i)
		fmt.Printf("Accuracy: %%%v", float64(successCount)/float64(totalProcessesed))
	}
}

func parseFile(filename string) (imgArrays [][][]int, expectedValues []int) {
	// Open the file
	file, err := os.Open(filename) // just pass the file name
	if err != nil {
		check(err)
	}
	defer file.Close()

	// Initailize the array of image bitmaps
	imgArrays = [][][]int{}

	// Start scanning the file line by line
	scanner := bufio.NewScanner(file)
	tempImgArray := [][]int{}
	tempExpectedValue := 0
	for scanner.Scan() {
		// Read in one line of the file
		tempLine := scanner.Text()

		// If we have a blank line then we finished a block of stuff
		// Save and scan until we get to then next block
		if tempLine == "" {
			imgArrays = append(imgArrays, tempImgArray)
			expectedValues = append(expectedValues, tempExpectedValue)

			tempImgArray = [][]int{}
			tempExpectedValue = 0

			// Skip one line and then go back to the beginning of the loop
			scanner.Scan()
			continue
		}

		// COnvert the row into an array of ints
		tempStrArray := strings.Split(tempLine, " ")

		// Line starts with a space, skip it
		if tempStrArray[0] == "" {
			tempStrArray = tempStrArray[1:]
		} else if tempStrArray[len(tempStrArray)-1] == "" {
			tempStrArray = tempStrArray[:len(tempStrArray)-1]
		}

		// Convert the row to ints
		rowInts := make([]int, len(tempStrArray))
		for i, elem := range tempStrArray {
			rowInts[i], err = strconv.Atoi(elem)

			if err != nil {
				panic(fmt.Errorf("failed to parse char into int: %v", elem))
			}
		}

		// We have 29 elements, we need to grab the expected value from the row
		if len(rowInts) == 29 {
			tempExpectedValue = rowInts[0]

			rowInts = rowInts[1:]
		}

		// Ad this row to the current bitmap
		tempImgArray = append(tempImgArray, rowInts)
	}

	if err := scanner.Err(); err != nil {
		check(err)
	}

	return imgArrays, expectedValues
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}
