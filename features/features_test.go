package features

import (
	"reflect"
	"testing"
)

var testBitmap_1 = [][]int{[]int{0, 0, 0, 0}, []int{0, 0, 0, 0}, []int{0, 0, 0, 0}, []int{0, 0, 0, 0}}
var testBitmap_2 = [][]int{[]int{1, 1, 1, 1}, []int{0, 0, 0, 0}, []int{0, 0, 0, 0}, []int{0, 0, 0, 0}}
var testBitmap_3 = [][]int{[]int{0, 0, 0, 1, 0},
	[]int{0, 1, 0, 1, 0},
	[]int{0, 1, 0, 1, 0},
	[]int{0, 1, 0, 1, 0},
	[]int{0, 0, 0, 0, 0}}

func TestRegion_1(t *testing.T) {
	actual := RegionAvg(testBitmap_3)

	expected := []float64{0, 0, 0, 1, 0,
		0, 1, 0, 1, 0,
		0, 1, 0, 1, 0,
		0, 1, 0, 1, 0,
		0, 0, 0, 0, 0}

	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("Error occured while testing Density: '%v' != '%v'", expected, actual)
	}
}

func TestDensity_1(t *testing.T) {
	actual := Density(testBitmap_1)

	expected := 0.0

	if actual != expected {
		t.Errorf("Error occured while testing Density: '%v' != '%v'", expected, actual)
	}
}

func TestDensity_2(t *testing.T) {
	actual := Density(testBitmap_2)

	expected := .25

	if actual != expected {
		t.Errorf("Error occured while testing Density: '%v' != '%v'", expected, actual)
	}
}

func TestVertSymmetry_1(t *testing.T) {
	actual := VertSymmetry(testBitmap_1)

	expected := 1.0

	if actual != expected {
		t.Errorf("Error occured while testing VertSymmetry: '%v' != '%v'", expected, actual)
	}
}

func TestVertSymmetry_2(t *testing.T) {
	actual := VertSymmetry(testBitmap_2)

	expected := .5

	if actual != expected {
		t.Errorf("Error occured while testing VertSymmetry: '%v' != '%v'", expected, actual)
	}
}

func TestHorizontalSymmetry_1(t *testing.T) {
	actual := HorizontalSymmetry(testBitmap_1)

	expected := 1.0

	if actual != expected {
		t.Errorf("Error occured while testing HorizontalSymmetry: '%v' != '%v'", expected, actual)
	}
}

func TestHorizontalSymmetry_2(t *testing.T) {
	actual := HorizontalSymmetry(testBitmap_2)

	expected := 1.0

	if actual != expected {
		t.Errorf("Error occured while testing HorizontalSymmetry: '%v' != '%v'", expected, actual)
	}
}

func TestHorizontalIntercept(t *testing.T) {
	actual_1, actual_2 := HorizontalIntercepts(testBitmap_3)

	expected_1 := 1
	expected_2 := 2

	if actual_1 != expected_1 {
		t.Errorf("Error occured while testing HorizontalIntercept(min): '%v' != '%v'", expected_1, actual_1)
	}

	if actual_2 != expected_2 {
		t.Errorf("Error occured while testing HorizontalIntercept(max): '%v' != '%v'", expected_2, actual_2)
	}
}

func TestVertIntercept(t *testing.T) {
	actual_1, actual_2 := VertIntercepts(testBitmap_3)

	expected_1 := 1
	expected_2 := 1

	if actual_1 != expected_1 {
		t.Errorf("Error occured while testing VertIntercept(min): '%v' != '%v'", expected_1, actual_1)
	}

	if actual_2 != expected_2 {
		t.Errorf("Error occured while testing VertIntercept(max): '%v' != '%v'", expected_2, actual_2)
	}
}
