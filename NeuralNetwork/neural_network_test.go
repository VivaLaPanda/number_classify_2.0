package NeuralNetwork

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestAll(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	nn := NewNeuralNetwork(1, 20, 1)

	//Train (y=sin(x))
	for i := 0; i < 1000000; i++ {
		tmpRandom := math.Pi * rand.Float64()
		nn.Backpropagation([]float64{tmpRandom}, []float64{math.Sin(tmpRandom)}, 0.7, 0.000001)
	}

	//Run Test
	fmt.Println("  Ans    :    NN")
	for i := 0; i < 20; i++ {
		var tmp float64
		tmp = math.Pi * float64(i) / 20.0
		outVec, err := nn.Calc([]float64{tmp})
		if err != nil {
			fmt.Println(err.Error())
			return
		}
		fmt.Printf("%f : %f\n", math.Sin(tmp), (outVec[0]))
	}
}
