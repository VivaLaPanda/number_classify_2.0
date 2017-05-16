// Package NeuralNetwork is a simple implementation of a three layer neural network
package NeuralNetwork

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// Makes neuron used to apply signmoid function
type SigmoidalNeuron struct {
	inputSignal  float64
	outputSignal float64
}

// Contructor for sigmoid NewSigmoidalNeuron
// Returns: A valid sigmoid neuron
func NewSigmoidalNeuron() *SigmoidalNeuron {
	return &SigmoidalNeuron{0, 0}
}

// Sigmoid's the signal coming into the neuron.
// Also modifies the neuron's state to include the new input and output
// Expects: A real number input
// Returns: A real number resulting from applying a sigmoid to the input
func (n *SigmoidalNeuron) calc(input float64) float64 {
	n.inputSignal = input
	n.outputSignal = n.sigmoid(input)
	return n.outputSignal
}

// A simple function which just passes a given input through the neuron
// without a signmoid. The neuron's state is still modified
// Expects: A real number input
// Returns: The input
func (n *SigmoidalNeuron) through(input float64) float64 {
	n.outputSignal = input
	n.inputSignal = input
	return input
}

// The function to be used in calculating sigmoids
// Expects: A real number to apply the function to
// Returns: The real number resulting from the sigmoid function
func (n *SigmoidalNeuron) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp2(-x))
}

// A simple type of error used for incorrect dimension sizes
type DimError struct {
	What string
}

// Prints out a dimension error.
func (e *DimError) Error() string {
	return fmt.Sprintf("at %s", e.What)
}

// The struct which encodes the neural NeuralNetwork
// We decided to use matrices instead of actual nodes for ease of computing
type NeuralNetwork struct {
	inputDim, hiddenDim, outputDim          int
	inputNeuron, hiddenNeuron, outputNeuron []SigmoidalNeuron
	wih, who                                [][]float64 // Weight vectors
}

// Simply generates a 2D slice of the given dimensions filled with random
// 64 bit floats
// Expects: Two integers indicating the lengths of the desired slice
// Returns: A two dimensional random slice with the given lengths
func NewRandom2dSlice(x int, y int) [][]float64 {
	s := make([][]float64, x)
	for i := 0; i < x; i++ {
		s[i] = make([]float64, y)
		for j := 0; j < y; j++ {
			s[i][j] = rand.Float64() - 0.5
		}
	}
	return s
}

// Constructor for the neural NeuralNetwork
// Expects:
//  inputNum - Number of inputs
//  hiddenNum - Number of neurons in the hidden layer
//  outputNum - Number of neurons in the output layer
// Returns: A valid neural network instance
func NewNeuralNetwork(inputNum int, hiddenNum int, outputNum int) *NeuralNetwork {
	wih := NewRandom2dSlice(inputNum+1, hiddenNum)
	who := NewRandom2dSlice(hiddenNum+1, outputNum)
	return &NeuralNetwork{inputNum, hiddenNum, outputNum,
		make([]SigmoidalNeuron, inputNum+1),
		make([]SigmoidalNeuron, hiddenNum+1),
		make([]SigmoidalNeuron, outputNum),
		wih, who}
}

// Calculates the output vector which results from a valid input
// Expects: A slice of length nn.inputDim
// Returns: A slice of length nn.outputDim
func (nn *NeuralNetwork) Calc(inputVec []float64) ([]float64, error) {
	if len(inputVec) != nn.inputDim {
		return nil, &DimError{fmt.Sprintf("Input error (a: %v, e: %v)", inputVec, nn.inputDim)}
	}

	// Getting inputs
	for i := 0; i < nn.inputDim; i++ {
		nn.inputNeuron[i].calc(inputVec[i])
	}
	nn.inputNeuron[nn.inputDim].through(1.0) // Bias neuron

	// Weighted sum of input neurons into each hidden neuron
	var sum float64
	for h := 0; h < nn.hiddenDim; h++ {
		sum = 0.0
		for i := 0; i < nn.inputDim+1; i++ {
			sum += nn.inputNeuron[i].outputSignal * nn.wih[i][h]
		}
		nn.hiddenNeuron[h].calc(sum)
	}

	nn.hiddenNeuron[nn.hiddenDim].through(1.0)

	// Weighted sum of hidden neurons into each output neuron
	for o := 0; o < nn.outputDim; o++ {
		sum = 0.0
		for h := 0; h < nn.hiddenDim+1; h++ {
			sum += nn.hiddenNeuron[h].outputSignal * nn.who[h][o]
		}
		nn.outputNeuron[o].calc(sum)
	}
	outputVec := make([]float64, nn.outputDim)
	for j := 0; j < nn.outputDim; j++ {
		outputVec[j] = nn.outputNeuron[j].outputSignal
	}
	return outputVec, nil
}

// implementation of Backpropagation algorithm
// Expects:
//  Vector of inputs to the network of length nn.inputDim
//  Vector which one want to recieve (from training data) of length nn.outputDim
//  Learning constant which should be determined through experimentation
//  Tolerance indicating when training should be stopped. If you don't wish
//    to use this parameter, just set it to 0
// Returns: Nothing, only changes state.
func (nn *NeuralNetwork) Backpropagation(inputVec []float64, targetVec []float64, learningConst float64, tolerance float64) error {
	if nn.inputDim != len(inputVec) {
		return &DimError{"inputVec dim error"}
	}
	if nn.outputDim != len(targetVec) {
		return &DimError{"targetVec dim error"}
	}
	outputVec, _ := nn.Calc(inputVec)

	convergeState := true
	for k := 0; k < nn.outputDim; k++ {
		diff := targetVec[k] - outputVec[k]
		if diff > tolerance || diff < -tolerance {
			convergeState = false
		}
	}
	if convergeState == true {
		return nil
	}

	// backpropagation
	// https://en.wikipedia.org/wiki/Backpropagation#Code
	deltaO := make([]float64, nn.outputDim)
	deltaH := make([]float64, nn.hiddenDim)
	tmpPreWho := make([][]float64, nn.hiddenDim+1)
	for i := 0; i < nn.hiddenDim+1; i++ {
		tmpPreWho[i] = make([]float64, nn.outputDim+1)
	}

	for k := 0; k < nn.outputDim; k++ {
		deltaO[k] = (targetVec[k] - outputVec[k]) * outputVec[k] * (1.0 - outputVec[k])
		oWg := &sync.WaitGroup{}
		oWg.Add(nn.hiddenDim + 1)
		for j := 0; j < nn.hiddenDim+1; j++ {
			go func(j int, k int) {
				defer oWg.Done()
				tmpPreWho[j][k] = nn.who[j][k]
				nn.who[j][k] += learningConst * deltaO[k] * nn.hiddenNeuron[j].outputSignal
			}(j, k)
		}

		oWg.Wait()
	}

	hWg := &sync.WaitGroup{}
	hWg.Add(nn.hiddenDim)
	for j := 0; j < nn.hiddenDim; j++ {
		go func(j int) {
			defer hWg.Done()
			deltaH[j] = 0.0
			for k := 0; k < nn.outputDim; k++ {
				deltaH[j] += deltaO[k] * tmpPreWho[j][k]
			}
			deltaH[j] *= nn.hiddenNeuron[j].outputSignal * (1.0 - nn.hiddenNeuron[j].outputSignal)
			for i := 0; i < nn.inputDim+1; i++ {
				nn.wih[i][j] += learningConst * deltaH[j] * nn.inputNeuron[i].outputSignal
			}
		}(j)
	}
	hWg.Wait()

	return nil
}
