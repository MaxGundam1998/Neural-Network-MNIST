#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

__device__ float relu(float x){
    return max(0.0f, x);
}

//Derivative of ReLu for backprop
__device__ float d_relu(float x){
    if (x <= 0){
        return 0.0f;
    }
    
    else{
        return 1.0f;
    }
}

__device__ void softmax(float* input, float* A2, int input_len){
    float sum = 0;
    float m = input[0];

    //Calculate the maximum value
    for(int i = 1; i < input_len; i++){
        m = max(m, input[i]);
    }

    for(int j = 0; j < input_len; j++){
        sum += exp(input[j] - m);
    }

    for(int k = 0; k < input_len; k++){
        A2[k] = exp(input[k] - m)/sum;
    }

}

__device__ void hot_one(float* label, float* Y_A, int label_index, int output_N){
    float Y = label[label_index];

    //printf("Value: %d\n", Y);

    for(int i = 0; i < output_N; i ++){
        if(i == Y){
            Y_A[i] = 1.0;
        }
        else{
            Y_A[i] = 0.0;
        }
    }

}

//Summation function
__device__ float summation(float* d_arr, int tid, int size){
    float sum = 0.0f;
    for(int i = 0; i < size; i++){
        float y = d_arr[i];

        //printf("%f ", y);

        float temp = sum + y;

        sum = temp;
        //printf("Current Sum: %f\n", sum);
    }
    
    //printf("End\n");
    return sum;
}

__global__ void print_Array(float* array, int size){
    for(int i = 0; i < size; i++){
        printf("%f ", array[i]);
    }
    printf("\n");
}


__global__ void forward_prop(float* input, float* W1, float* Z1, float* B1, float* A1, float* W2, float* Z2, float* B2, float* A2, int input_N, int hidden_N, int output_N, int num_samples){
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if(tid < num_samples){
        
        //Z1 = W1*input + B1
        for(int i = 0; i < hidden_N; i++){
            //Add the Biases
            Z1[tid * hidden_N + i] = B1[i];

            //Multiply Inputs with weights and add it to the Z1
            for(int j = 0; j < input_N; j++){
                Z1[tid * hidden_N + i] += input[tid * input_N + j] * W1[j * hidden_N + i];
            }
        }

        //A1 = Relu(Z1)
        for(int i = 0; i < hidden_N; i++){
            A1[tid * hidden_N + i] = relu(Z1[tid * hidden_N + i]);
        }

        //Z2 = W2 * A1 + B2
        for(int i = 0; i < output_N; i++){
            Z2[tid * output_N + i] = B2[i];

            for(int j = 0; j < hidden_N; j++){
                Z2[tid * output_N + i] += A1[tid * hidden_N + j] * W2[j * output_N + i];
            }

        }

        //A2 = Softmax(Z2)
        softmax(Z2, A2, output_N);
    }
}


__global__ void back_prop(float* host_data, float* A2, float* Z2, float* W2, float* B2, float* deZ2, float* deB2, float* deW2, float* A1, float* Z1, float* W1, float* B1, float* deZ1, float* deB1, float* deW1, float* labels, float learn, int e, int num_samples, int batch_num, int batch_size, int input_N, int hidden_N, int output_N){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int label_index = tid + (batch_num * e);

    float* Y_A = (float*)malloc(output_N*sizeof(float));

    //Initialize values to 0
    for(int i = 0; i < output_N; i++){
        Y_A[i] = 0.0f;
    }

    hot_one(labels, Y_A, label_index, output_N);

    //dZ2 = A2 - Y_A
    for(int i = 0; i < output_N; i++){
        deZ2[tid * output_N + i] = A2[tid * output_N + i] - Y_A[tid * output_N + i]; 
    }

    //dW2 = (1/m)*(Dz2) * A2(Transposed)
    for(int i = 0; i < output_N; i++){
        for(int j = 0; j < hidden_N; j++){
            deW2[tid * hidden_N * output_N + j * output_N + i] = deZ2[tid * output_N + i] * A1[tid * hidden_N + j];
        }
    }

    //dB2 = (1/m) * summation(dz2)
    for(int i = 0; i < output_N; i++){
        float sum = 0.0;
        for(int j = 0; j < batch_size; j++){
            sum += deZ2[j * output_N + i];
        }
        deB2[i] = sum / batch_size;
    }

    //dZ1 = W2(transposed) * dZ2 
    for(int i = 0; i < hidden_N; i++){
        for(int j = 0; j < output_N; j++){
            deZ1[tid * hidden_N + i] += W2[i * output_N + j] * deZ2[tid * output_N + j];
        }
        deZ1[tid * hidden_N + i] *= d_relu(Z1[tid * hidden_N + i]);
    }

    //dW1 = (1/m)*dZ1 * inputs(Transposed)
    for(int i = 0; i < hidden_N; i++){
        for(int j = 0; j < input_N; j++){
            float sum = 0.0;
            for(int k = 0; k < batch_size; k++){
                sum += deZ1[k * hidden_N + i] * host_data[k * input_N + j];
            }
            deW1[j * hidden_N + i] = sum / batch_size;
        }
    }

    //dB1
    for(int i = 0; i < hidden_N; i++){
        float sum = 0.0;
        for(int j = 0; j < batch_size; j++){
            sum += deZ1[j * hidden_N + i];
        }
        deB1[i] = sum /batch_size;
    }
    //Update all the values

    //W1 = W1 - learn(dW1)
    for(int i = tid; i < (hidden_N*input_N); i += blockDim.x * gridDim.x){
        W1[tid * (hidden_N*input_N) + i] -= learn*deW1[tid * (hidden_N*input_N) + i];
    }

    //B1 = B1 - learn(dB1)
    for(int i = tid; i < hidden_N; i += blockDim.x * gridDim.x){
        B1[tid *hidden_N + i] -= learn*deB1[tid * hidden_N + i];
    }

    //W2 = W2 - learn(dW2)
    for(int i = tid; i < (output_N*hidden_N); i += blockDim.x * gridDim.x){
        W2[tid * (output_N*hidden_N) + i] -= learn*deW2[tid * (output_N*hidden_N) + i];
    }

    //B2 = B2 - learn(dB2)
    for(int i = tid; i < output_N; i += blockDim.x * gridDim.x){
        B2[tid * output_N + i] -= learn*deB2[tid * output_N + i];
    }


}

__device__ int prediction(float* A2, int output_N){
    //Perform argmax on A2
    int max_index = 0;
    float max_val = A2[0];

    for(int i = 1; i < output_N; i++){
        if(A2[i] > max_val){
            max_val = A2[i];
            max_index = i;
        }
    }

    return max_index;
}

__global__ void accuracy(){
    //Get the Predictions first
    //Then Calculate the accuracy
}


int main(){
    FILE *fp;
    char filename[] = "Data/mnist_train.csv";

    //Declare arrays to store data
    //Arrays to store MNIST data
    float **data;
    float **pixel_data;
    float *labels;

    //Data that will be used for calculation
    float *host_data;
    float *W1;
    float *W2;
    float *B1;
    float *B2;
    float *Z1;
    float *Z2;
    float *A1;
    float *A2;

    //GPU Values
    float *d_host_data;
    float *d_W1;
    float *d_W2;
    float *d_B1;
    float *d_B2;
    float *d_Z1;
    float *d_Z2;
    float *d_A1;
    float *d_A2;

    float *d_labels;

    //Values for Backwards Propagtion
    //dZ2, dW2, dB2, dZ1, dW1, dB1
    float* deZ2;
    float* deW2;
    float* deB2;
    float* deZ1;
    float* deW1;
    float* deB1;

    float* d_deZ2;
    float* d_deW2;
    float* d_deB2;
    float* d_deZ1;
    float* d_deW1; 
    float* d_deB1;

    //Size of Layers
    const int input = 784;
    const int hidden = 10;
    const int output = 10;

    //One extra for the rows and identifiers
    const int ROWS = 60000;
    const int COLS = 785;

    //Open file
    fp = fopen(filename, "r");
    if(fp == NULL){
        printf("Error opening file.\n");
        return 1;
    }

    //Allocate memory for array to store MNIST Data
    data = (float**)malloc(ROWS * sizeof(float*));
    for(int i = 0; i < ROWS; i++){
        data[i] = (float*)malloc(COLS * sizeof(float));
        
    }

    pixel_data = (float**)malloc((COLS - 1)*sizeof(float*));
    for(int i = 0; i < (COLS - 1); i++){
        pixel_data[i] = (float*)malloc(ROWS * sizeof(float));
    }

    labels = (float*)malloc(ROWS * sizeof(float *));

    //Fill the Allocated array with the values read from the mnist_train.csv file.
    char line[10000];
    int row = 0;
    while(fgets(line, sizeof(line), fp) != NULL && row < 60000){
        char* token = strtok(line, ",");
        int col = 0;
        while(token != NULL && col < 785){
            data[row][col] = atof(token);
            token = strtok(NULL, ",");
            //printf("Label: %f\n", data[row][col]);
            col++;
        }
        row++;
    }

    //Grab Label Data
    for(int i = 0; i < ROWS; i++){
        labels[i] = data[i][0];
    }

    //Transpose the data
    //784 x 60000 Matrix
    for (int i = 0; i < COLS - 1; i++) {
        for (int j = 0; j < ROWS; j++) {
        pixel_data[i][j] = data[j][i+1];
        //printf("%f\n", pixel_data[i][j]);
        }
    }

    //Flattened version of the transposed array into the host_data array
    host_data = (float*)malloc(((COLS-1)*ROWS) * sizeof(float));
    for(int i = 0; i < 784; i++){
        for(int j = 0; j < 60000; j++){
            int position = i*60000 + j;
            host_data[position] = pixel_data[i][j];
        }
    }

    //------------------------------------------------------------------------------------------------------------------------------
    
    //Declare weight arrays
    //W1 size of input size times hidden size = 7840
    //W2 size of hidden size times output size = 100
    W1 = (float*)malloc((hidden*input)*sizeof(float));
    W2 = (float*)malloc((output*hidden)*sizeof(float));
    deW1 = (float*)malloc((hidden*input)*sizeof(float));
    deW2 = (float*)malloc((output*hidden)*sizeof(float));
    
    //Declare Biases 
    B1 = (float*)malloc(hidden*sizeof(float));
    B2 = (float*)malloc(output*sizeof(float));
    deB1 = (float*)malloc(hidden*sizeof(float));
    deB2 = (float*)malloc(output*sizeof(float));

    //Declare Z value arrays
    Z1 = (float*)malloc(hidden*sizeof(float));
    Z2 = (float*)malloc(output*sizeof(float));
    deZ1 = (float*)malloc(hidden*sizeof(float));
    deZ2 = (float*)malloc(output*sizeof(float));

    //Declare A value arrays
    A1 = (float*)malloc(hidden*sizeof(float));
    A2 = (float*)malloc(output*sizeof(float));

    //Random Num generator
    float random_W1;
    float random_B1;
    float random_W2;
    float random_B2;
    srand(time(NULL));

    //Assign random values to W1, W2, B1 and B2
    for(int i = 0; i < (hidden*input); i++){
        random_W1 = ((float)rand() / (float)RAND_MAX) -  0.5;
        W1[i] = random_W1;
    }
    
    for(int i = 0; i < (output*hidden); i++){
        random_W2 = ((float)rand() / (float)RAND_MAX) - 0.5;
        W2[i] = random_W2;
    }

    for(int i = 0; i < hidden; i++){
        random_B1 = ((float)rand() / (float)RAND_MAX) - 0.5;
        B1[i] = random_B1;
    }

    for(int i = 0; i < output; i++){
        random_B2 = ((float)rand() / (float)RAND_MAX) - 0.5;
        B2[i] = random_B2;
    }

    //-------------------------------------------------------------------------------------------------------------------

    //Fill Z1, Z2, deZ1, deZ2, deB1, deB2, deW1, deW2, A1 and A2 with zeroes
    for(int i = 0; i < hidden; i++){
        Z1[i] = 0.0f;
        A1[i] = 0.0f;
        deZ1[i] = 0.0f;
        deB1[i] = 0.0f;
    }

    for(int i = 0; i < output; i++){
        Z2[i] = 0.0f;
        A2[i] = 0.0f;
        deZ2[i] = 0.0f;
        deB2[i] = 0.0f;
    }

    for(int i = 0; i < (input*hidden); i++){
        deW1[i] = 0.0f;
    }

    for(int i = 0; i < (output*hidden); i++){
        deW2[i] = 0.0f;
    }

    //-----------------------------------------------------------------------------------------------------------------
    //Create Gpu values
    const size_t bytes_input = ((COLS-1) * ROWS) * sizeof(float);
    const size_t bytes_Labels = ROWS * sizeof(float);
    const size_t bytes_W1 = (hidden*input) * sizeof(float);
    const size_t bytes_W2 = (output*hidden) * sizeof(float);
    const size_t bytes_B1 = hidden * sizeof(float);
    const size_t bytes_B2 = output * sizeof(float);
    const size_t bytes_Z1 = hidden * sizeof(float);
    const size_t bytes_Z2 = output * sizeof(float);
    const size_t bytes_A1 = hidden * sizeof(float);
    const size_t bytes_A2 = output * sizeof(float);


    //Assign memory values
    cudaMalloc(&d_host_data, bytes_input);
    cudaMalloc(&d_labels, bytes_Labels);
    cudaMalloc(&d_W1, bytes_W1);
    cudaMalloc(&d_B1, bytes_B1);
    cudaMalloc(&d_A1, bytes_A1);
    cudaMalloc(&d_Z1, bytes_Z1);
    cudaMalloc(&d_W2, bytes_W2);
    cudaMalloc(&d_B2, bytes_B2);
    cudaMalloc(&d_A2, bytes_A2);
    cudaMalloc(&d_Z2, bytes_Z2);

    //Backwards Prop Val
    cudaMalloc(&d_deZ2, bytes_Z2);
    cudaMalloc(&d_deW2, bytes_W2);
    cudaMalloc(&d_deB2, bytes_B2);
    cudaMalloc(&d_deZ1, bytes_Z1);
    cudaMalloc(&d_deW1, bytes_W1);
    cudaMalloc(&d_deB1, bytes_B1);

    //Memory copy for gpu
    cudaMemcpy(d_host_data, host_data, bytes_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, bytes_Labels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, bytes_W1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, B1, bytes_B1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A1, A1, bytes_A1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z1, Z1, bytes_Z1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, bytes_W2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, B2, bytes_B2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z2, Z2, bytes_Z2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A2, A2, bytes_A2, cudaMemcpyHostToDevice);


    //cudaMemcpy backward values
    cudaMemcpy(d_deZ2, deZ2, bytes_Z2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_deW2, deW2, bytes_W2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_deB2, deB2, bytes_B2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_deZ1, deZ1, bytes_Z1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_deW1, deW1, bytes_W1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_deB1, deB1, bytes_B1, cudaMemcpyHostToDevice);

    //Begin Training
    int epoch = 10;
    float batch_size = 64;
    float num_samples = 60000;
    float learn_rate = 0.1;

    float batch_num = round(num_samples / batch_size);
    //printf("Batch Num: %f\n", batch_num);

    int batch_numI = (int)batch_num;

    int num_blocks = (num_samples + batch_size - 1) / batch_size;
    //printf("Num blocks: %d\n", num_blocks);

    dim3 block_size(batch_size, 1, 1);
    dim3 grid_size(num_blocks, 1, 1);
    

    //Epochs: Refers to the complete pass of a training dataset through the neural network model
    //Batch: Refers to a subset of the training data that is processed together in one forward and backward pass
    //Ten epochs and 64 batch size
    //Number of batches 938
    for(int e = 0; e < epoch; e++){
        //printf("Epoch %d\n", e + 1);
        for(int i = 0; i < batch_numI; i++){

            //Forward Propagation
            forward_prop<<<grid_size, block_size>>>(d_host_data, d_W1, d_Z1, d_B1, d_A1, d_W2, d_Z2, d_B2, d_A2, input, hidden, output, num_samples);

            //Backward Propagation and update vals in function
            back_prop<<<grid_size, block_size>>>(d_host_data, d_A2, d_Z2, d_W2, d_B2, d_deZ2, d_deB2, d_deW2, d_A1, d_Z1, d_W1, d_B1, d_deZ1, d_deB1, d_deW1, d_labels, learn_rate, e, num_samples, batch_numI, batch_size, input, hidden, output);

        }
        //Print the Accuracy.
        //predictions(A2);
    }

    //Free GPU memory
    cudaFree(d_host_data);
    cudaFree(d_labels);
    cudaFree(d_Z1);
    cudaFree(d_W1);
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_Z2);
    cudaFree(d_W2);
    cudaFree(d_B2);

    cudaFree(deZ2);
    cudaFree(deW2);
    cudaFree(deB2);
    cudaFree(deZ1);
    cudaFree(deW1);
    cudaFree(deB1);

    //Free CPU memory
    for(int i = 0; i < ROWS; i++){
        free(data[i]);
    }

    for(int i = 0; i < COLS - 1; i++){
        free(pixel_data[i]);
    }

    free(W1);
    free(Z1);
    free(A1);
    free(B1);
    free(W2);
    free(Z2);
    free(A2);
    free(B2);
    free(host_data);
    free(labels);

    fclose(fp);


    printf("End of code");

}
