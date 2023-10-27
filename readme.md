# PANN - postnerd's artificial neural network
Just a plain JavaScript implementation of a perceptron.

## Install

```
npm install @postnerd/pann
```

## Usage

```javascript
import PANN from "pann";

const pann = new PANN();

// train network
const trainData = [
    [[0.47,0.57],1],
    [[0.78,-1],0],
    [[-0.43,-0.5],1],
    [[-0.72,0.32],1],
    [[-0.03,0.25],1],
    [[-0.19,0.41],1],
    [[0.74,-0.51],0],
    [[-0.72,0.85],1],
    [[0.62,-0.15],1],
    [[-0.96,-0.84],1],
    [[0.46,-0.44],1],
    [[-0.89,0.22],1],
    [[0.4,0.04],1],
    [[-0.58,-0.84],1],
    [[-0.48,-0.22],1],
    [[-0.21,0.89],1],
    [[-0.5,1],1],
    [[-0.09,0.52],1],
    [[-0.9,-0.69],1],
    [[-0.53,-0.27],1]
]

trainData.forEach((dataSet) => {
    const dataSet = dataSet[0];
    const expectetResult = dataSet[1];
    pann.train(dataSet, expectedResult);
});

// predict
const set = [[0.3, 0.8], 1];
const result = pann.predict(set[0], set[1]);

console.log(`Data set: ${JSON.stringify(set[0])} Expected result: ${set[1]} Result: ${result}`);

// see weights and bias
const weightAndBias = pann.getWeightAndBias();

// see data for last run
const lastRun = pann.getDataForLastRun();

// see full data
const fullData = pann.getDataForFullHistory();
```
