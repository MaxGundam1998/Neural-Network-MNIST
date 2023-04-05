#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_S 784;
#define HS 10;
#define OUTPUT_S 10;
#define LEARN 0.1;

__global__ void print2DArray(float* array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if(array[i*cols +j] != 0)
            {
                printf("%f\n", array[i * cols + j]);
            }
            else continue;
        }
    }
}

__global__ void print_Array(float* array, int size){
    for(int i = 0; i < size; i++){
        printf("%f ", array[i]);
    }
    printf("\n");
}

__global__ void print_ArrayNO0(float* array, int size){
    int count = 0;
    int C = 0;
    for(int i = 0; i < size; i++){
        if(array[i] == 0){
            count += 1;
        }
        else
            C += 1;
        
    }
    printf("Zeroes %d\n", count);
    printf("Non-zeroes %d\n", C);
}

__global__ void printInt(int* array, int size){
    for(int i = 0; i < size; i++){
        printf("%d ", array[i]);
    }
    printf("\n");
}

//Using Kahan summation algorithm
__device__ float summation(float* d_arr, int size){
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

//Relu activation function
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

    //Printing debugger
    /*for(int i = 0; i < output_N; i++){
        printf("%f ", Y_A[i]);
    }*/

}

void columnPrint(float* f_arr, int col, int row, int c_index){
    for(int i = 0; i < row; i++){
        printf("%f\n", f_arr[i*col + c_index]);
    }
}

__global__ void CP(float* f_arr, int col, int row, int c_index){
    for(int i = 0; i < row; i++){
        printf("%f\n", f_arr[i*col + c_index]);
    }
}

//Forward Propagation
__global__ void forward_propagation(float* input_data, float* weights1, float* biases1, float* weights2, float* biases2, 
                                    float* A1, float* A2, float* Z2, float* Z1, float* output_data, int num_samples, int input_N, int hidden_N1, int output_N)
{
     //net_input += input_data[i] * weights1[i * hidden_N1 + tid];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < num_samples){
        //Z1 = W1*input + B1
        for(int i = 0; i < hidden_N1; i++){
            //Add the Biases
            Z1[tid * hidden_N1 + i] = biases1[i];

            //Multiply Inputs with weights and add it to the Z1
            for(int j = 0; j < input_N; j++){
                Z1[tid * hidden_N1 + i] += input_data[tid * input_N + j] * weights1[j * hidden_N1 + i];
            }
        }

        //Use Relu
        for(int i = 0; i < hidden_N1; i++){
            A1[tid * hidden_N1 + i] = relu(Z1[tid * hidden_N1 + i]);
        }

        //Z2 = A1*W2 + B2
        for(int i = 0; i < output_N; i++){
            Z2[tid * output_N + i] = biases2[i];
            for(int j = 0; j < hidden_N1; j++){
                Z2[tid * output_N + i] += A1[tid * hidden_N1 + j] * weights2[j * output_N + i];
            }
        }


    }
    //A2 = Softmax(Z2)
    softmax(Z2, A2, output_N);

    for(int i = 0; i < output_N; i++){
        output_data[tid * output_N + i] = A2[i];
    }
}

//This needs work.
//Backwards Propagation
__global__ void backward_propagation(float* g_output_data, float* g_input_data, float* dZ2, float* dW2, float* A1, float* dB2, 
                                        float* W2, float* dZ1, float* Z1, float* dW1, float* dB1, float* labels, float* Y_A, int input_N, int output_N, int hidden_N1, int rows, int m)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < m)
    {
        hot_one(labels, Y_A, tid, output_N);

        //dZ2 = g_output_data - Y_A
        for(int i = 0; i < output_N; i++){
            
            dZ2[i] = g_output_data[i] - Y_A[i];
            //printf("g_output_data - Y_A = dz2\n");
            //printf("%f - %f = %f\n", g_output_data[i], Y_A[i], dZ2[i]);
        }

        //dW2 = (1/m)*dZ2 * A1Transposed
        for(int i = 0; i < output_N; i++){
            for(int j = 0; j < hidden_N1; j++){
            dW2[i*hidden_N1+j] = (1/m) * dZ2[i] * A1[j];
            }
        }

        //Calculate summation of Dz2
        float sumDZ2 = 0.0f;
        sumDZ2 = summation(dZ2, output_N);

        //db2 = (1/m) * summation(dz22);
        for(int i = 0; i < output_N; i++){
            dB2[i] = (1.0/m) * sumDZ2;
        }

        //dz1 = W2(transposed)*dz2 
        for(int i = 0; i < hidden_N1; i++){
        float sum = 0.0f;
            for(int j = 0; j < output_N; j++){
                sum += W2[i + j*hidden_N1] * dZ2[j] * d_relu(Z1[i]);
            }
            dZ1[i] = sum;
        }

        //dw1 = (1/m)dz1 inputs(transposed)
        //Fixed
        for(int i = 0; i < hidden_N1; i++){
            for(int j = 0; j < input_N; j++){
                dW1[i*input_N + j] = (1/m) * dZ1[i] * g_input_data[j * rows + tid]; 
                //printf("%f * %f = %f\n", dZ1[i], g_input_data[j * rows + tid], dW1[i*input_N + j]);
            }
        }

        //db1 = (1/m) * summation(dZ1)
        float sumDZ1 = 0.0f;

        sumDZ1 = summation(dZ1, hidden_N1);

        for(int i = 0; i < hidden_N1; i++){
            dB1[i] = (1.0/m) * sumDZ1;
        }

    }

}

__global__ void update(float* W1, float* dW1, float* B1, float* dB1, float* W2, float *dW2, float* B2, float* dB2, float learn, int m, int output_N, int hidden_N, int input_N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < m){
        //Update W1
        //Total amount of weights (hidden x input)
        for(int i = 0; i < (hidden_N*input_N); i++){
            W1[i] = W1[i] - (learn)*dW1[i];
            //printf("%f \n", dW1[i]);
        }

        //Update B1
        for(int i = 0; i < hidden_N; i++){
            B1[i] = B1[i] - (learn) * dB1[i];
        }

        //Update W2
        for(int i = 0; i < (hidden_N * output_N); i++){
            W2[i] = W2[i] - (learn) * dW2[i];
        }

        //Update B2
        for(int i = 0; i < output_N; i++){
            B2[i] = B2[i] - (learn) * dB2[i];
        }
    }


}

int main(){
    
    //Define Max Cols 785
    //Grab the mnist_training data
    FILE *fp;
    char filename[] = "Data/mnist_train.csv";

    //Declare arrays to store data
    float **data;
    float **pixel_data;
    float *labels;

    //One extra for the rows and identifiers
    int ROWS = 60000;
    int COLS = 785;

    //Open file
    fp = fopen(filename, "r");
    if(fp == NULL){
        printf("Error opening file.\n");
        return 1;
    }

    //Allocate memory for array of data
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
        //printf("Label: %d\n", labels[i]);
    }

    //Transpose the data somehow.
    //784 x 60000 Matrix
    for (int i = 0; i < COLS - 1; i++) {
        for (int j = 0; j < ROWS; j++) {
        pixel_data[i][j] = data[j][i+1];
        //printf("%f\n", pixel_data[i][j]);
        }
    }

    //Flattened version of the transposed array.
    float* Host_Data = (float*)malloc(((COLS-1)*ROWS) * sizeof(float));

    for(int i = 0; i < 784; i++){
        for(int j = 0; j < 60000; j++){
            int position = i*60000 + j;
            Host_Data[position] = pixel_data[i][j];
        }
    }

    //---------------------------------------------------------------------------------------------------------------------------------

    //784 INPUT Values for the MNIST data
    const int input_N = 784;
 
    //10 Output values
    const int output_N = 10;

    //10 Nodes in Hidden Layer 1
    const int hidden_N1 = 10;

    //Initialize random weights and Biases
    //Using Relu Function, so weights are initialize for ReLu
    
    float **weights1;
    float **weights2;
    float *biases1;
    float *biases2;
    float *output_data;

    weights1 = (float **)malloc(hidden_N1 * sizeof(float *));
    for(int i = 0; i < hidden_N1; i++){
        weights1[i] = (float *)malloc(input_N * sizeof(float));
    }

    weights2 = (float **)malloc(output_N * sizeof(float *));
    for(int i = 0; i < output_N; i++){
        weights2[i] = (float *)malloc(hidden_N1 * sizeof(float));
    }


    biases1 = (float *)malloc(hidden_N1 * sizeof(float *));
    biases2 = (float *)malloc(output_N * sizeof(float *));

    output_data = (float *)malloc(output_N * sizeof(float));


    //Create the weights. Values will be -0.5 to 0.5
    float random_num;
    float random_bia;
    srand(time(NULL));

    //Input Layer to Hidden layer weights
    for(int i = 0;  i < hidden_N1; i++)
    {
        for(int j = 0; j < input_N; j++){
            random_num = ((float)rand() / (float)RAND_MAX) -  0.5;
            weights1[i][j] = random_num;
            //printf("%f\n", weights1[i][j]);
        }
    }

    //Weights for Hidden to Output
    for(int i = 0; i < output_N; i++)
    {
        for(int j = 0; j < hidden_N1; j++){
            random_num = ((float)rand() / (float)RAND_MAX) -  0.5;
            weights2[i][j] = random_num;
            //printf("%d, %d\n", i, j);
            //printf("%f\n", weights1[i][j]);
        }
    }

    //flat weights matrix 1
    float flatW1[hidden_N1*input_N];
    for(int i = 0; i < hidden_N1; i++){
        for(int j = 0; j < input_N; j++){
            int position = i*input_N + j;
            flatW1[position] = weights1[i][j];
        }
    }

    //Flat weight Matrix 2
    float flatW2[hidden_N1*output_N];
    for(int i = 0; i < output_N; i++){
        for(int j = 0; j < hidden_N1; j++){
            int position = i*hidden_N1 + j;
            flatW2[position] = weights2[i][j];
        }
    }

    //Assigning Bias one values
    for(int i = 0; i < hidden_N1; i++)
    {
        random_bia = ((float)rand() / (float)RAND_MAX) - 0.5;
        biases1[i] = random_bia;
        //printf("%f\n", biases1[i]);
    }

    //Assigning Bias two values
    for(int i = 0; i < output_N; i++){
        random_bia = ((float)rand() / (float)RAND_MAX) - 0.5;
        biases2[i] = random_bia;
    }

    //Intialize output array to zero
    for(int i = 0; i < output_N; i++){
        output_data[i] = 0.0f;
    }
     
    //Declare Z1, A1, Z2, A2
    float* Z1;
    float* A1;
    float* Z2;
    float* A2;

    //GPU dz1 and dz2
    float* dZ1;
    float* dZ2;

    Z1 = (float*)malloc(hidden_N1 * sizeof(float));
    A1 = (float*)malloc(hidden_N1 * sizeof(float));
    Z2 = (float*)malloc(output_N * sizeof(float));
    A2 = (float*)malloc(output_N * sizeof(float));

    dZ2 = (float*)malloc(output_N * sizeof(float));
    dZ1 = (float*)malloc(hidden_N1 * sizeof(float));

    //Fill them with zeroes
    for(int i = 0; i < hidden_N1; i++){
        Z1[i] = 0.0f;
        A1[i] = 0.0f;
        dZ1[i] = 0.0f;
    }

    for(int i = 0; i < output_N; i++){
        Z2[i] = 0.0f;
        A2[i] = 0.0f;
        dZ2[i] = 0.0f;
    }

    //Backwards Variables
    float* dW2;
    dW2 = (float*)malloc((output_N * hidden_N1) * sizeof(float));
    for(int i = 0; i < (output_N * hidden_N1); i++){
        dW2[i] = 0.0f;
    }

    float* dW1;
    dW1 = (float*)malloc((hidden_N1 * input_N) * sizeof(float));
    for(int i = 0; i < (hidden_N1 * input_N); i++){
        dW1[i] = 0.0f;
    }

    float* dB2;
    dB2 = (float*)malloc(output_N * sizeof(float));
    for(int i = 0; i < output_N; i++){
        dB2[i] = 0.0f;
    }

    float* dB1;
    dB1 = (float*)malloc(hidden_N1 * sizeof(float));
    for(int i = 0; i < hidden_N1; i++){
        dB1[i] = 0.0f;
    }

    float* Y_A;
    Y_A = (float*)malloc(output_N * sizeof(float));
    for(int i = 0; i < output_N; i++){
        Y_A[i] = 0.0f;
    }
    
    //-----------------------------------------------------------------------------------------------------

    //The learning rate
    //float learn = 0.1f;

    //Use Cuda Malloc Pitch for 2D arrays: Inputs, weights1 and weights2
    float* g_input_data;
    const size_t bytes_input = ((COLS - 1)*ROWS) * sizeof(float);

    float* g_weights1;
    const size_t bytes_W1 = (hidden_N1*input_N)*sizeof(float);

    float* g_weights2;
    const size_t bytes_W2 = (output_N * hidden_N1)*sizeof(float);

    float* g_biases1;
    const size_t byte_B1 = hidden_N1*sizeof(float);

    float* g_biases2;
    const size_t byte_B2 = output_N*sizeof(float);

    float* g_output_data;
    const size_t byte_out = output_N*sizeof(float);

    float* g_A1;
    const size_t bytes_A1 = hidden_N1*sizeof(float);

    float* g_Z1;
    const size_t bytes_Z1 = hidden_N1*sizeof(float);

    float* g_A2;
    const size_t bytes_A2 = output_N*sizeof(float);

    float* g_Z2;
    const size_t bytes_Z2 = output_N*sizeof(float);

    float* g_labels;
    const size_t bytes_labels = ROWS*sizeof(float);

    //Backwards Propagation
    float* g_dZ2;
    const size_t bytes_dZ2 = output_N * sizeof(float);

    float* g_dW2;
    const size_t bytes_dW2 = (output_N * hidden_N1) * sizeof(float);

    float* g_dB2;
    const size_t bytes_dB2 = output_N * sizeof(float);

    float* g_dZ1;
    const size_t bytes_dZ1 = hidden_N1 * sizeof (float);

    float* g_dB1;
    const size_t bytes_dB1 = hidden_N1 * sizeof(float);

    float* g_dW1;
    const size_t bytes_dW1 = (hidden_N1 * input_N) * sizeof(float);

    float* g_Y_A;
    const size_t bytes_Y_A = output_N * sizeof(float);

    cudaMalloc(&g_input_data, bytes_input);
    cudaMalloc(&g_weights1, bytes_W1);
    cudaMalloc(&g_weights2, bytes_W2);
    cudaMalloc(&g_biases1, byte_B1);
    cudaMalloc(&g_biases2, byte_B2); 
    cudaMalloc(&g_A1, bytes_A1);
    cudaMalloc(&g_Z1, bytes_Z1);
    cudaMalloc(&g_A2, bytes_A2);
    cudaMalloc(&g_Z2, bytes_Z2);
    cudaMalloc(&g_output_data, byte_out);
    cudaMalloc(&g_labels, bytes_labels);
  

    //For Backward's propagation
    cudaMalloc(&g_dZ2, bytes_dZ2);
    cudaMalloc(&g_dW1, bytes_dW1);
    cudaMalloc(&g_dB2, bytes_dB2);
    cudaMalloc(&g_dZ1, bytes_dZ1);
    cudaMalloc(&g_dB1, bytes_dB1);
    cudaMalloc(&g_dW2, bytes_dW2);
    cudaMalloc(&g_Y_A, bytes_Y_A);

    //printf("Did it work?\n");

    //Values for forward Propagation
    cudaMemcpy(g_input_data, Host_Data, bytes_input, cudaMemcpyHostToDevice);
    cudaMemcpy(g_weights1, flatW1, bytes_W1, cudaMemcpyHostToDevice);
    cudaMemcpy(g_weights2, flatW2, bytes_W2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_biases1, biases1, byte_B1, cudaMemcpyHostToDevice);
    cudaMemcpy(g_biases2, biases2, byte_B2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_A1, A1, bytes_A1, cudaMemcpyHostToDevice);
    cudaMemcpy(g_Z1, Z1, bytes_Z1, cudaMemcpyHostToDevice);
    cudaMemcpy(g_A2, A2, bytes_A2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_Z2, Z2, bytes_Z2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_output_data, output_data, byte_out, cudaMemcpyHostToDevice);

    //Cuda Memcpy for backwards
    cudaMemcpy(g_dZ2, dZ2, bytes_dZ2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dB1, dB1, bytes_dB1, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dW1, dW1, bytes_dW1, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dB2, dB2, bytes_dB2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dZ1, dZ1, bytes_Z1, cudaMemcpyHostToDevice);
    cudaMemcpy(g_Y_A, Y_A, bytes_Y_A, cudaMemcpyHostToDevice);
    

    //print2DArray<<<1, 1>>>(g_input_data, 10, 784);
    //print_Array<<<1, 1>>>(g_input_data, (784*60000));
    int block_size = 256;
    int grid_size = (input_N + block_size - 1) / block_size;
    int m = ROWS;
    int epochs = 10;

    float learn = 0.1;

    for(int i = 0; i < epochs; i++){

    }


    /*
    //Perform Forward propagation
    forward_propagation<<<grid_size, block_size>>>(g_input_data, g_weights1, g_biases1, g_weights2, g_biases2, g_A1, g_A2, g_Z2, g_Z1, g_output_data, m, input_N, hidden_N1, output_N);

    //Call Label data before after forward and before backward
    cudaMemcpy(g_labels, labels, bytes_labels, cudaMemcpyHostToDevice);


    backward_propagation<<<grid_size, block_size>>>(g_output_data, g_input_data, g_dZ2, g_dW2, g_A1, g_dB2, g_weights2, g_dZ1, g_Z1, g_dW1, g_dB1, g_labels, g_Y_A, input_N, output_N, hidden_N1, ROWS, m);
     

    //Update Parameters
    update<<<grid_size, block_size>>>(g_weights1, g_dW1, g_biases1, g_dB1, g_weights2, g_dW2, g_biases2, g_dB2, learn, m, output_N, hidden_N1, input_N);
    */

    //Freeing the data
    //Of the arrays
    for(int i = 0; i < ROWS; i++){
        free(data[i]);
    }

    for(int i = 0; i < COLS - 1; i++){
        free(pixel_data[i]);
    }

    for(int i = 0; i < hidden_N1; i++){
        free(weights1[i]);
    }

    for(int i = 0; i < output_N; i++){
        free(weights2[i]);
    }

    free(pixel_data);
    free(data);
    free(labels);
    free(weights1);
    free(weights2);
    free(biases1);
    free(biases2);
    free(A1);
    free(A2);
    free(Z1);
    free(Z2);
    free(output_data);
    //free(L);

    //Free Backwards
    free(dW1);
    free(dZ2);
    free(dB2);
    free(dZ1);
    free(dB1);
    free(dW2);
    free(Y_A);

    // Free GPU memory
    cudaFree(g_input_data);
    cudaFree(g_weights1);
    cudaFree(g_biases1);
    cudaFree(g_weights2);
    cudaFree(g_biases2);
    cudaFree(g_A1);
    cudaFree(g_A2);
    cudaFree(g_Z1);
    cudaFree(g_Z2);
    cudaFree(g_output_data);
    cudaFree(g_labels);
    //cudaFree(g_L);

    //Free backwards gpu
    cudaFree(g_dW1);
    cudaFree(g_dZ2);
    cudaFree(g_dB2);
    cudaFree(g_dZ1);
    cudaFree(g_dW2);
    cudaFree(g_dB1);
    cudaFree(g_Y_A);

    fclose(fp);
    
    printf("You've reached the end\n");

}
