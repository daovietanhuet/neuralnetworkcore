class Matrix {
    constructor(rows, cols){
        this.rows = rows;
        this.cols = cols;
        this.data = [];
        for(let i = 0; i < this.rows; i++) {
            this.data[i] = [];
            for(let j = 0; j < this.cols; j++){
                this.data[i][j] = 0;
            }
        }
    }

    randomize = () => {
        for(let i = 0; i < this.rows; i++) {
            for(let j = 0; j < this.cols; j++){
                this.data[i][j] = Math.random()*2 - 1;
            }
        }
    }

    static op = (m, op) => {
        let result_matrix = new Matrix(m[0].rows, m[0].cols);
        for(let i = 0; i < m[0].rows; i++) {
            for(let j = 0; j < m[0].cols; j++){
                if(m.length === 1)
                    result_matrix.data[i][j] = op(m[0].data[i][j]);
                else if (m.length === 2)
                    result_matrix.data[i][j] = op(m[0].data[i][j], m[1].data[i][j]);
                else if (m.length === 3)
                    result_matrix.data[i][j] = op(m[0].data[i][j], m[1].data[i][j], m[2].data[i][j]);
                else {
                    throw new Error('maximum number of element')
                }
            }
        }
        return result_matrix;
    }

    static dot = (m, n) => {
        if(n instanceof Matrix && m instanceof Matrix){
            if(m.cols !== n.rows)
                throw new Error('Can\'t dot product two matrix')
            let result_matrix = new Matrix(m.rows, n.cols);
            for(let i = 0; i < m.rows; i++){
                for(let j = 0; j < n.cols; j ++){
                    for(let k = 0; k < m.cols; k ++){
                        result_matrix.data[i][j] += m.data[i][k]*n.data[k][j];
                    }
                }
            }
            return result_matrix;
        } else {
            throw new Error('m or n is invalid')
        }
    }

    static T = (n) => {
        let result_matrix = new Matrix(n.cols, n.rows);
        for(let i = 0; i < n.rows; i++) {
            for(let j = 0; j < n.cols; j++){
                result_matrix.data[j][i] = n.data[i][j];
            }
        }
        return result_matrix;
    }


    copy = () => {
        let result_matrix = new Matrix(this.rows, this.cols);
        for(let i = 0; i < this.rows; i++) {
            for(let j = 0; j < this.cols; j++){
                result_matrix.data[i][j] = this.data[i][j];
            }
        }
        return result_matrix;
    }

    static fromArray = (input) => {
        let m = new Matrix(input.length, 1);
        for(let i = 0; i < input.length; i++){
            m.data[i][0] = input[i];
        }
        return m;
    }

    print = () => {
        console.table(this.data)
    }

    serialize() {
        return JSON.stringify(this);
    }
    
    static deserialize(data) {
        if (typeof data == 'string') {
          data = JSON.parse(data);
        }
        let matrix = new Matrix(data.rows, data.cols);
        matrix.data = data.data;
        return matrix;
    }
}

sigmoid = (x1, x2) => 1 / (1 + Math.exp(-(x1+x2)));
cal_gradient = (x1, x2) => (x1*(1-x1))*x2;
error = (x1, x2) => -(x1/x2 - (1-x1)/(1-x2));
loss_function = (x1, x2) => -(x1*Math.log(x2) + (1-x1)*Math.log(1-x2));

class NeuralNetwork {
    constructor(init, dest_loss){
        this.layers = init;
        this.learning_rate = 0.1;
        this.dest_loss = dest_loss;
        this.w = [];
        this.b = [];

        for(let i = 0 ; i < this.layers.length - 1; i++) {
            this.w[i] = new Matrix(this.layers[i+1], this.layers[i]);
            this.w[i].randomize();
            this.b[i] = new Matrix(this.layers[i+1], 1);
        }
    }

    predict = (input_arr) => {
        let X = Matrix.fromArray(input_arr);
        for(let i = 0; i < this.layers.length - 1; i ++) {
            X = Matrix.op([Matrix.dot(this.w[i], X), this.b[i]], sigmoid);
        }
        return X.serialize()
    }

    trainning = (trainning_data) => {
        while (true) {
            let data = trainning_data[Math.floor(Math.random()*100)%data_trainning.length];
            let A = [];
            A[0] = Matrix.fromArray(data.input);
            let y = Matrix.fromArray(data.target);
            // feed forward
            for(let i = 0; i < this.layers.length - 1; i ++) {
                A[i+1] = Matrix.op([Matrix.dot(this.w[i], A[i]), this.b[i]], sigmoid);
            }
            // backpropagation
            let dA = [];
            dA[0] = Matrix.op([y, A[A.length - 1]], error);
            let dw = [];
            let db = [];
            for(let i = this.layers.length - 2; i >= 0; i --) {
                let gradient = Matrix.op([A[i+1], dA[dA.length - 1]], cal_gradient);
                dw[i] = Matrix.dot(gradient, Matrix.T(A[i]));
                db[i] = gradient;
                dA[i] = Matrix.dot(Matrix.T(this.w[i]), gradient) ;
            }
            // gradient descent
            for(let i = 0; i < this.layers.length - 1; i ++) {
                this.w[i] = Matrix.op([this.w[i], dw[i]], (x1, x2) => x1 - x2*this.learning_rate);
                this.b[i] = Matrix.op([this.b[i], db[i]], (x1, x2) => x1 - x2*this.learning_rate);
            }
            // loss function
            let loss = Matrix.op([y, A[A.length - 1]], loss_function);
            let result_loss = 0;
            for(let i = 0; i < loss.data.length; i ++) {
                result_loss += loss.data[i][0];
            }
            if(Math.random() < 0.001) {
                console.log('accuracy >>>',1 - result_loss);
            }
            if(result_loss < this.dest_loss) break;
        }
    }

    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(data) {
        if (typeof data == 'string') {
            data = JSON.parse(data);
        }
        let nn = new NeuralNetwork(data.layers);
        for(let i = 0; i < data.layers.length - 1; i ++) {
            nn.w[i] = Matrix.deserialize(data.w[i]);
            nn.b[i] = Matrix.deserialize(data.b[i]);
        }
        nn.learning_rate = data.learning_rate;
        return nn;
    }
}
