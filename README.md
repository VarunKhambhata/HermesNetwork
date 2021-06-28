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
  <br>
  
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
  <br>
  
  ```c++
   struct Layer
  ```
  ###### This structure is a handle to single layer in a neural network. An object of struct NeuralNetwork  consist of linked list of struct Layer. Layer also contain additional informations like no. of neurons, no. of weights, pointers to next and previous Layer and layerType.
  
  ```c++
  enum layerType
  ```
  ###### It contain 3 value: inputL, outputL, hiddenL. A value of this enum is stored in struct Layer object, which gives necessary information to the framework for doing operations.
  
  ```c++
  Layer* initLayer(int size, layerType typ);
  ```
  ###### Creates Layer object of given type and size and return pointer of that object. It executes necessary OpenGL procedures to creates array of neurons inside GPU. FIrst parameter is the size which is no. of neurons in the layer and second parameter is layerType which can be either of inputL, outputL, hiddenL.
  
  ```c++
  NeuralNetwork* createNetwork(int InputSize, int OutputSize);
  ```
  ###### Creates a NeuralNetwork object and return it pointer. There are no hidden layer in the network. It only consist of input layer and output layer of specified size.
  
  ```c++
  void connectLayer(NeuralNetwork* Network, Layer* prev, Layer* next);
  ```
  ###### Inserts a layer next to specified previous layer. After connecting, it generates necessary weights and bias between the two layer. A new layer can also be inserted between previously connected layers. First paramenteris a pointer of NeuralNetwork, second parameter is pointer of Layer after which a given layer will be inserted, third paremeter is pointer of Layer which will be inserted.
  
  ```c++
  float* getWeights_Bias(NeuralNetwork* Network, int LayerDepth);
  ```
  ###### Returns an array of all the weights of layer at the specified depth. If layer at specified depth has 2 neurons amd layer before it has 3 neuron, the array returned will have data as: |w|w|w|b|w|w|w|b| where w is wieght valule and b is bias value; first 4 array elements belongs to first neuron and last four array elemtnets belongs to the second neuron.
  
  ```c++
  void triggerLayer(Layer* Lyr);
  ```
  ###### Activates all neurons of a given layer. It calculates weighted sum of inputs from previous layer and use activation function to generate output value. NOTE: do not trigger input layer as it receives data sent not from the previous layer and there is no previous layer from input layer.
  
  ```c++
  float* getLayerNeuronsData(NeuralNetwork* Network, int LayerDepth);
  ```
  ###### Returns array of data of every neuron of layer at specified depth.
  
  ```c++
  void calcError(Layer* Lyr, float* ActualOutput);
  ```
  ###### Calculate error of each neurons in a layer. Error value are stored inside the neuron aside from activation value.
  
  ```c++
  void backPropogateError(Layer* Lyr);
  ```
  ###### Get errors from next layer neurons and backpropogate those error to the given layer.
  
  ```c++
  void trainLayer(Layer* Lyr, float* LearningRate);
  ```
  ###### Update the weight of the given layer using error value generated and given LearningRate.
  ```c++
  -Shader Codes-
  const char* vertexShader_code;
  const char* sigmoidActivationShader_code;
  const char* WeightInitShader_code;
  const char* WeightUpdateShader_code;
  const char* ErrorGen_code;
  const char* ErrorBackPropogate_code;  
  ```
  ###### All these string contains OpenGL shader program which are used to actually run neural network in a GPU. These programs are compiled when `initNeuralLink()` is called.
  
</details>
  
