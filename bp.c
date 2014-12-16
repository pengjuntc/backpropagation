#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define DATASIZE 800  //size of training data and test data
#define ETA 0.01   //learning rate
#define BIASOUTPUT 1   //output for bias. It's always 1.
#define MAX_EPOCHS 50000   //maximum epochs 

//**********************struct for dataset**********************************//
typedef struct ENTRY{   //dataset for each sample data
  double x1;
  double x2;
  int desired_value;
} ENTRY;


//**********************struct for NEURON**********************************//
typedef struct NEURON{   
  double input;
  double output;
  double *weights;
  double delta;
  double error;
} NEURON;

//**********************struct for LAYER***********************************//
typedef struct LAYER{
  int numberOfNeurons;
  NEURON *neurons;
} LAYER;

//*********************struct for NNET************************************//
typedef struct NNET{
  double *inputs;
  int numberOfLayers;
  LAYER *layers;
} NNET; //neural network


//********sigmoid function and randomWeight generator********************//

double sigmoid(double v)
{
  return 1/(1+ exp(-v));
}

double randomWeight()  //random weight generator between -0.5 ~ 0.5 
{
  return ((int)rand()%100000)/(float) 100000 - 0.5;
}

//****************************create neuron network*********************//

void createNeuronNetwork(NNET *net, int numberOfLayers, int *neuronsOfLayer )
{ 
  //in order to create a neural network, 
  //we need know how many layers and how many neurons in each layer

  int i, j, k;
  srand(time(NULL));
  net->numberOfLayers = numberOfLayers;
  
  assert(numberOfLayers >= 3);
  
  net->layers = (LAYER *)malloc(numberOfLayers*sizeof(LAYER));
  //construct input layer, no weights
  net->layers[0].numberOfNeurons = neuronsOfLayer[0];
  net->layers[0].neurons = (NEURON *) malloc (neuronsOfLayer[0] * sizeof(NEURON));

  //construct hidden layers
  for(i = 1; i < numberOfLayers; i++)    //construct layers 
    {                                              
      net->layers[i].neurons = (NEURON *) malloc (neuronsOfLayer[i] * sizeof(NEURON));
      net->layers[i].numberOfNeurons = neuronsOfLayer[i];
      for(j = 0; j < neuronsOfLayer[i]; j++)  // construct each neuron in the layer
	{
	  net->layers[i].neurons[j].weights = (double *)malloc((neuronsOfLayer[i-1]+1)*sizeof(double));
	  for(k = 0; k <= neuronsOfLayer[i-1]; k++)  
	    {    //construct weights of neuron from previous layer neurons
	      net->layers[i].neurons[j].weights[k] = randomWeight(); //when k = 0, it's bias weight
	      //net->layers[i].neurons[j].weights[k] = 0;
	    }
	}
    }
}

//****************************feed forward***************************//

double feedforward(NNET *net, ENTRY data)   //return the root of mean square error
{
  //set the output of input layer
  //two inputs x1 and x2
  net->layers[0].neurons[0].output = data.x1;
  net->layers[0].neurons[1].output = data.x2;

  int i, j, k;
  for(i = 1; i < net->numberOfLayers; i++)  //calculate output from hidden layers to output layer
    {
      for(j = 0; j < net->layers[i].numberOfNeurons; j++)
	{
	  double v = 0;  //induced local field for neurons
	  //calculate v, which is the sum of the product of input and weights 
	  for(k = 0; k <= net->layers[i-1].numberOfNeurons; k++)
	    {  
	      if(k == 0)
		v+=net->layers[i].neurons[j].weights[k]*BIASOUTPUT;
	      else
		v+=net->layers[i].neurons[j].weights[k]*net->layers[i-1].neurons[k-1].output;
	    }

	  //double output = sigmoid(v);
	  //net->layers[i].neurons[j].output = output;
	  net->layers[i].neurons[j].output = sigmoid(v);
	  //printf("%lf\n", output);
	}
    }

  //calculate mean square error;
  double y[4] = {0,0,0,0};   //desired value array
  y[data.desired_value-1] = 1;
  double sumOfSquareError = 0;
  int numberOfLayers = net->numberOfLayers;
  for(i = 0; i < net->layers[numberOfLayers-1].numberOfNeurons; i++)
    {
      //error = desired_value - output
      double error = y[i] - net->layers[numberOfLayers-1].neurons[i].output; 
      net->layers[numberOfLayers-1].neurons[i].error = error;
      sumOfSquareError += error*error/2;
    }
  double mse = sumOfSquareError / net->layers[numberOfLayers-1].numberOfNeurons;
  return mse;   //return the root of mean square error
}


//**************************backpropagation***********************//

void backpropagation(NNET *net)
{
  //calculate delta
  int i, j, k;
  int numberOfLayers = net->numberOfLayers;
  
  //calculate delta for output layer
  for(i = 0; i < net->layers[numberOfLayers-1].numberOfNeurons; i++)
    {
      double output = net->layers[numberOfLayers-1].neurons[i].output;
      double error = net->layers[numberOfLayers-1].neurons[i].error;
      //for output layer, delta = y(1-y)error
      net->layers[numberOfLayers-1].neurons[i].delta = output*(1-output)*error;
    }

  //calculate delta for hidden layers
  for(i = numberOfLayers - 2; i > 0; i--)
    {
      for(j = 0; j < net->layers[i].numberOfNeurons; j++)
	{
	  double output = net->layers[i].neurons[j].output;
	  double sum = 0;
	  for(k = 0 ; k < net->layers[i+1].numberOfNeurons; k++)
	    {
	      sum += net->layers[i+1].neurons[k].weights[j+1]*net->layers[i+1].neurons[k].delta;
	    }
	  net->layers[i].neurons[j].delta = output*(1-output)*sum;
	}
    }

  //update weights
  for(i = 1; i < numberOfLayers; i++)
    {
      for(j = 0; j < net->layers[i].numberOfNeurons; j++)
	{
	  for(k = 0; k <= net->layers[i-1].numberOfNeurons; k++)
	    {
	      double inputForThisNeuron;
	      if(k == 0)
		inputForThisNeuron = 1;  //bias input
	      else
		inputForThisNeuron = net->layers[i-1].neurons[k-1].output;

	      net->layers[i].neurons[j].weights[k] += ETA*net->layers[i].neurons[j].delta*inputForThisNeuron;
	    }
	}
    }
}


//*******************read training data and testing data**********************//

void getTrainingAndTestData(int argc, char **path, ENTRY *training, ENTRY *testing)
{
  if(argc != 3)
    {
      printf("Usage: program training_data_file testing_data_file\n");
      exit(0);
    }

  FILE *fp1, *fp2;
  if((fp1 = fopen(path[1], "r")) == NULL)
    {
      printf("cannot open %s\n", path[1]);
      exit(1);
    }
  if((fp2 = fopen(path[2], "r")) == NULL)
    {
      printf("cannot open %s\n", path[2]);
      exit(1);
    }

  int i = 0;
  int num1, num2;
  while(i < 800)
   {
     fscanf(fp1, "%d %d %lf %lf", &num1, &num2, &training[i].x1, &training[i].x2);
     fscanf(fp2, "%d %d %lf %lf", &num1, &num2, &testing[i].x1, &testing[i].x2);
     if(i < 200 ) 
       {
	 training[i].desired_value = 1;
	 testing[i].desired_value = 1;
       }
     else if(i < 400)
       {
	 training[i].desired_value = 2;
	 testing[i].desired_value = 2;
       }
     else if(i < 600)
       {
	 training[i].desired_value = 3;
	 testing[i].desired_value = 3;
       }
     else 
       {
	 training[i].desired_value = 4;
	 testing[i].desired_value = 4;
       }
     i++;
   }
  fclose(fp1);
  fclose(fp2);
}


//********shuffle the order of presentation to neuron*********//
void swap(ENTRY *data, int i, int j)
{
  ENTRY temp;
  temp.x1 = data[i].x1;
  temp.x2 = data[i].x2;
  temp.desired_value = data[i].desired_value;
  data[i].x1 = data[j].x1;
  data[i].x2 = data[j].x2;
  data[i].desired_value = data[j].desired_value;
  data[j].x1 = temp.x1;
  data[j].x2 = temp.x2;
  data[j].desired_value = temp.desired_value;
}

void shuffle(ENTRY *data, int size)
{
  srand(time(NULL));
  int i;
  for(i = 0; i < size; i++)
    {
      int j = (int)rand()%size;
      swap(data, i, j);
    }
}


/*
void printWeights(NNET *net)
{
  int i, j, k;
   for(i = 1; i < net->numberOfLayers; i++)   
    {                     
      printf("Layer %d\n", i);
      for(j = 0; j < net->layers[i].numberOfNeurons; j++)
	{
	  printf("\t Neuron %d : ", j);
	  for(k = 0; k <= net->layers[i-1].numberOfNeurons; k++)  
	    {
	      printf("%lf ", net->layers[i].neurons[j].weights[k]);
	    }
	  printf("\n");
	}
    }
}
*/


//*************************calculate error average*************//

double relativeError(double *error, int len)
{
  len = len - 1;
  if(len < 20)
    return 1;
  //keep track of the last 20 Root of Mean Square Errors
  int start1 = len-20;
  int start2 = len-10;

  double error1 = 0;
  double error2 = 0;
  
  int i;
  
  //calculate the average of the first 10 errors
  for(i = start1; i < start1 + 10; i++)
    {
      error1 += error[i];
    }
  double averageError1 = error1 / 10;

  //calculate the average of the second 10 errors
  for(i = start2; i < start2 + 10; i++)
    {
      error2 += error[i];
    }
  double averageError2 = error2 / 10;
  
  double relativeErr = (averageError1 - averageError2)/averageError1;
  return (relativeErr > 0) ? relativeErr : -relativeErr;
}



//**************************main function***********************//

int main(int argc, char** argv)
{
  NNET *Net =(NNET *)malloc(sizeof(NNET));
  int numberOfLayers = 3;
  //the first layer -- input layer
  //the last layer -- output layer
  int neuronsOfLayer[3] = {2, 3, 4}; 
  ENTRY *training_data = (ENTRY *)malloc(DATASIZE*sizeof(ENTRY));
  ENTRY *testing_data = (ENTRY *)malloc(DATASIZE*sizeof(ENTRY));
  
  //read training data and testing data from file
  getTrainingAndTestData(argc, argv, training_data, testing_data);  
  
  //create neural network for backpropagation
  createNeuronNetwork(Net, numberOfLayers, neuronsOfLayer);
  
  //error array to keep track of errors
  double *error = (double *)malloc(MAX_EPOCHS * sizeof(double));
  int maxlen = 0;
  int i;
  int epoch = 1;

  //output data to a file
  FILE *fout;
  if((fout = fopen("randomtest_1.txt", "w")) == NULL)
    {
      fprintf(stderr, "file open failed.\n");
      exit(1);
    }
  
  //shuffle the order of presenting training data to neural network
  //shuffle(training_data, DATASIZE);
  
  //train and test neural network
  do{
    double squareErrorSum = 0;
    
    //shuffle the order of presenting training data to neural network
    //shuffle(training_data, DATASIZE);
    
    //train network
    for(i = 0; i < DATASIZE; i++)
      {
	squareErrorSum += feedforward(Net, training_data[i]);
	backpropagation(Net);
      }
    error[maxlen] = sqrt(squareErrorSum / DATASIZE);
    //test network
    double RMSForTest = 0;
    for(i = 0; i < DATASIZE; i++)
      {
	RMSForTest += feedforward(Net, testing_data[i]);
      }
    RMSForTest = sqrt(RMSForTest / DATASIZE);
    printf("%d %lf %lf\n", epoch, error[maxlen], RMSForTest);
    fprintf(fout, "%d %lf %lf\n", epoch, error[maxlen], RMSForTest);
    maxlen++;
    epoch++;
  }while(relativeError(error, maxlen) > 1e-4 && maxlen < MAX_EPOCHS);
  
  fclose(fout);
  free(Net);
  free(training_data);
  free(testing_data);
}
