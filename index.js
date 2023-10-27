export default class PANN {
    #weights = [];
    #bias;

    #history = {
        numberOfRuns: 0,
        dataSets: [],
        weightedSums: [],
        expectedResults: [],
        calcResults: [],
        errors: [],
        weights: [],
        bias: [],
    };

    static getRandomWeight() {
        const rand = Math.round(Math.random() * 100) / 100;
        return Math.random() > 0.5 ? rand : -rand;
    }

    constructor(numberOfInputs = 2) {
        for (let i = 0; i < numberOfInputs; i++) {
            this.#weights.push(PANN.getRandomWeight());
        }
        this.#bias = PANN.getRandomWeight();
    }

    #getAdjustedWeight(error = 0, value = 1) {
        return error * value;
    }

    #getWeightedSum(a, b) {
        let dotProduct = 0;
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
        }

        return dotProduct;
    }

    #activation(sum = 0) {
        if (sum >= 0) {
            return 1;
        }
        else {
            return 0;
        }
    }

    train(dataSet = [], expectedResult = 1) {
        const inputs = [...dataSet, 1];
        const weights = [...this.#weights, this.#bias];

        const weightedSum = this.#getWeightedSum(inputs, weights);
        const calcResult = this.#activation(weightedSum);
        const error = expectedResult - calcResult;

        this.#history.dataSets.push([...dataSet]);
        this.#history.weightedSums.push(weightedSum);
        this.#history.expectedResults.push(expectedResult);
        this.#history.calcResults.push(calcResult);
        this.#history.weights.push([...this.#weights]);
        this.#history.errors.push(error);
        this.#history.bias.push(this.#bias);
        this.#history.numberOfRuns++;

        if (error !== 0) {
            for (let i = 0; i < this.#weights.length; i++) {
                this.#weights[i] += this.#getAdjustedWeight(error, inputs[i]);
            }
            this.#bias = this.#getAdjustedWeight(error, 1);
        }
    }

    predict(dataSet = []) {
        const inputs = [...dataSet, 1];
        const weights = [...this.#weights, this.#bias];

        return this.#activation(this.#getWeightedSum(inputs, weights));
    }

    getWeightsAndBias() {
        return {
            weights: this.#weights,
            bias: this.#bias,
        };
    }

    getDataForFullHistory() {
        return this.#history;
    }

    getDataForLastRun() {
        let lastRun = this.#history.numberOfRuns - 1;

        return {
            numberOfRun: this.#history.numberOfRuns,
            dataSet: this.#history.dataSets[lastRun],
            weightedSum: this.#history.weightedSums[lastRun],
            expectedResult: this.#history.expectedResults[lastRun],
            calcResult: this.#history.calcResults[lastRun],
            error: this.#history.errors[lastRun],
            weights: this.#history.weights[lastRun],
            bias: this.#history.bias[lastRun],
        };
    }
}