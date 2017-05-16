package features

import "math"

// Takes the image (20 by 20) and divides it into a 5 by 5 grid where each
// square in the grid has 16 pixels. The average value of the 16
// pixels will become a feature. Thus each image will be converted to a vector
// of 25 real values.
func RegionAvg(imgArray [][]int) []float64 {
	yLen := len(imgArray) / 5
	xLen := len(imgArray[0]) / 5
	outputSlice := []float64{}

	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			regionSlice := make([][]int, len(imgArray))
			copy(regionSlice, imgArray[yLen*y:yLen*(y+1)])

			for i, row := range imgArray[yLen*y : yLen*(y+1)] {
				regionSlice[i] = row[xLen*x : xLen*(x+1)]
			}

			regionDensity := Density(regionSlice)

			outputSlice = append(outputSlice, regionDensity)
		}
	}

	return outputSlice
}

// Determines what percentage of the image
// is 1 pixels
func Density(imgArray [][]int) float64 {
	fgPx := 0.0
	bgPx := 0.0
	for _, row := range imgArray {
		for _, elem := range row {
			if elem == 0 {
				bgPx++
			} else {
				fgPx++
			}
		}
	}

	return fgPx / (bgPx + fgPx)
}

func VertSymmetry(imgArray [][]int) float64 {
	// Get total number of pixels
	// Assumes that array isn't ragged
	totalPx := float64(len(imgArray) + len(imgArray[0]))
	vertSymmetricPx := 0.0

	for i, row := range imgArray[:len(imgArray)/2] {
		for j, topElement := range row {
			bottomElement := imgArray[(len(imgArray)/2)+i][j]

			if topElement == bottomElement {
				vertSymmetricPx++
			}
		}
	}

	return vertSymmetricPx / totalPx
}

func HorizontalSymmetry(imgArray [][]int) float64 {
	// Get total number of pixels
	// Assumes that array isn't ragged
	totalPx := float64(len(imgArray) + len(imgArray[0]))
	horizontalSymmetricPx := 0.0

	for _, row := range imgArray {
		for j, leftElement := range row[:len(row)/2] {
			rightElement := row[(len(row)/2)+j]

			if leftElement == rightElement {
				horizontalSymmetricPx++
			}
		}
	}

	return horizontalSymmetricPx / totalPx
}

func HorizontalIntercepts(imgArray [][]int) (minIntercepts int, maxIntercepts int) {
	minIntercepts = math.MaxInt64
	maxIntercepts = math.MinInt64
	prevPx := 0

	for _, row := range imgArray {
		numIntercepts := 0

		for _, element := range row {
			if prevPx == 1 && element == 0 {
				numIntercepts++
			}

			prevPx = element
		}

		if numIntercepts < minIntercepts && numIntercepts != 0 {
			minIntercepts = numIntercepts
		}

		if numIntercepts > maxIntercepts {
			maxIntercepts = numIntercepts
		}
	}

	return minIntercepts, maxIntercepts
}

func VertIntercepts(imgArray [][]int) (minIntercepts int, maxIntercepts int) {
	minIntercepts = math.MaxInt64
	maxIntercepts = math.MinInt64
	prevPx := 0

	for i, row := range imgArray {
		numIntercepts := 0

		for j, _ := range row {
			if prevPx == 1 && imgArray[j][i] == 0 {
				numIntercepts++
			}

			prevPx = imgArray[j][i]
		}

		if numIntercepts < minIntercepts && numIntercepts != 0 {
			minIntercepts = numIntercepts
		}

		if numIntercepts > maxIntercepts {
			maxIntercepts = numIntercepts
		}
	}

	return minIntercepts, maxIntercepts
}
