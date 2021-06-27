# HermesNetwork
A C++ library using OpenGL 3 to build and run neural network on any GPU.<br>
No GPU configuration and selection is required as OpenGL drivers will handle all kind of GPUs.

This library has an API and a Framework.

### `HermesNetwork API` 
> The API part of library consist of publicaly avaliable functions that can be used in main program after including the header file.<br>
> API functions contains set of instuctions and calls to the HermesNetwork Framework, which provides an abstraction for building a conventional neural network.<br>
> Below is shown the content of HermesNetwork API:


<details>
  <summary>Click to view API document </summary>
  
  ```c++
  void InitNeuralLink(FunctionPointer GL_init_func);
  ```
  ###### This function must be called at the begining of main method. It setups gl context, compile shaders, create drawing polygon. As the library depends on OpenGL, OpenGL must be first initialized. Every OpenGL SDK has a function to initialize the library. This function name must be type casted to FunctionPointer and passed as an argument in this function. <br> If you are usnig GLEW for OpenGL SDK, this function can be called as `InitNeuralLink( (FunctionPointer)glewInit  );`

  ```c++
  struct NeuralNetwork
  ```
  ###### This structure is a handle to entire network. It must always be used as pointer object. It also contain additional informations like no. of layers, no. of inputs, no. of outputs and total no. of weights

  ```c++
  NeuralNetwork* NetworkBuilder(int InputSize, initializer_list<int> HiddenLayers, int OutputSize);
  ```
  ###### It builds the neural network inside GPU and return a pointer of NerualNetwork structure.<br> Fist argument is input layer size, second agrument is a list of sizes of hidden layer which must be written as "{s1, s2, s3, ... }". If there is no hidden layer, simpily specify empty list like "{ }". And third argument is the size of output layer.

  ```c++
  void AddLayer(NeuralNetwork* Network, int size, unsigned int Depth = -1);
  ```
  ###### This adds a new hidden layer at specified depth in the network. If depth is not specified, the new layer will be added just before the output layer.<br> First argument is the pointer object of NeuralNetwork struct, second argument is size of layer and third agrument is position of layer, which is optional.

  ```c++
  void SendInputs(NeuralNetwork* Network, float Inputs[]);
  ```
  ###### This function send the array of inputs to the input layer. <br> First argument is the pointer object of NeuralNetwork struct and second argurment is array of inputs.

  ```c++
  float* GetOutputLayerData(NeuralNetwork* Network);
  ```
  ###### It returns an array of float type of the data in output layer. <br> First argument is the pointer object of NeuralNetwork struct.


  ```c++
  void TriggerNetwork(NeuralNetwork* Network);
  ```
  ###### This function runs the network by activating each layers from input layer to output layer serialy and generates ouputs in ouput layer. 

  ```c++
  void TriggerLayer(NeuralNetwork* Network, int LayerDepth);
  ```
  ###### Activates only the layer located at specifed depth.

  ```c++
  void TrainNetwork(NeuralNetwork* Network, float ActualOutput[], float LearningRate = 1.0);
  ```
  ###### This function generates error in output layer, backpropogate errors to previous hidden layers and updates every weight and bias which in turn result in trainig of the network.
</details>
  
  
### `HermesNetwork Framework` 
> The Framework part of this library is written inside `HermesNetwork::` namespace.
> It contains functions and objects which allows more detailed operations of the neural network.
> Functions of this framework are also called from inside of the API functions.

<details>
  <summary>Click to view Framework document</summary>
  
  
</details>
