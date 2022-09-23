// *********************************************************************************** //
// Created by Varun Khambhata
// 2022.5.16
//
// HermesNetwork is made to harness any GPU with OpenGL driver to train and run
// nerural network without any external dependencies and can be integrated directly
// in your application.
//
// Requires OpenGL 4.2 driver (minimum)
// *********************************************************************************** //

#ifndef __HERMES_NETWORK__
#define __HERMES_NETWORK__

#include <initializer_list>
#include <fstream>
#include <ctime>

#ifdef _WIN32
    #include <windows.h>
#endif
#ifdef __linux__
    //include linux dependent header for creating window
#endif
#ifdef __APPLE__
    //include macOS dependent header for creating window
#endif
#ifdef __ANDROID__
    //include android dependent header for creating window
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////// DECLARATIONS ////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 *********************************************************
    This namespace is a framework which contains all 
    core vars and functions for HermesNetwork
 *********************************************************
*/
namespace HermesNetwork
{
    //////////////////////////////////////////// Objects ///////////////////////////////////////
    enum layerType	 {	inputL, outputL, hiddenL };
    enum networkType {	convolutional };

    //Handle a layer in network.
    struct LayerHandle
    {
        int no_neuron = 0;
        int no_weight = 0;
        layerType type;
        LayerHandle* next = nullptr;
        LayerHandle* prev = nullptr;
        unsigned int NeuronsTex = 0;
        unsigned int WeightsTex = 0;
        float* data = nullptr;
        float* weights = nullptr;
        int AFun = 0;
    };
    typedef LayerHandle* Layer;

    //Handle to entire network.    
    struct NeuralNetworkHandle
    {
        unsigned int no_layers = 2;
        unsigned int no_of_input = 0;
        unsigned int no_of_output = 0;
        // unsigned int total_weights = 0;
        HermesNetwork::networkType netType;
        HermesNetwork::Layer inputLayer = nullptr;
        HermesNetwork::Layer outputLayer = nullptr;
        GLuint pbo;
        float* Out = nullptr;
    };
    typedef NeuralNetworkHandle* NeuralNetwork;

    #ifdef _WIN32
        HWND offscreen_context;
        HGLRC GlRenderingContext;
    #endif
    #ifdef __linux__
        //Declare object of linux's window and rendering context
    #endif
    #ifdef __APPLE__
        //Declare object of macOS's window and rendering context
    #endif
    #ifdef __ANDROID__
        //Create object of android's window and rendering context
    #endif

    unsigned int TempTex;
    unsigned int WeightInit, Activation, ErrorGen, WeightUpdate, ErrorBackPropogate;
    int WINT_unifm_seed;
    int ACTV_unifm_prev_size, ACTV_unifm_weight_size, ACTV_unifm_prev_L_TEX, ACTV_unifm_Layer_weight;
    int WGHTUP_unifm_prev_size, WGHTUP_unifm_next_size, WGHTUP_unifm_LearnRT, WGHTUP_unifm_weight_TEX, WGHTUP_unifm_prev_L_TEX, WGHTUP_unifm_neuronOut_TEX;
    int ERROR_unifm_neuronOut_TEX, ERROR_unifm_actualOut_TEX;
    int ERROR_BP_unifm_neuronOut_TEX, ERROR_BP_unifm_next_L_TEX, ERROR_BP_unifm_weight_TEX, ERROR_BP_unifm_Layer_size, ERROR_BP_unifm_next_L_size;
    int ACTVLibs_unifm_SEL;

    ////////////////////////////////////////////// Functions /////////////////////////////////////////////////////////

    //This function will be called whenever a new layer is created
    Layer initLayer(int size, layerType typ);

    //Creates network with only input and output layer without any connecting weights. Returns a pointer of type- NeuralNetwork
    NeuralNetwork createBasicNetwork(int InputSize, int OutputSize);

    //Add a new hidden layer before the output layer
    void appendHiddenLayer(NeuralNetwork Network, int LayerSize);

    //Connect two layer with weight and initialize the weights
	void connectLayer(Layer prev, Layer next);

    //Activates every neurons of the given layer
	void triggerLayer(Layer Lyr);

    //Returns array of data of every neuron from given layer
	float* bindLayerNeuronsData(Layer Lyr);

    //Get neurons value in Layer's data array in CPU memory
	void fetchLayerNeuronsData(Layer Lyr);    

    //Get weights & bias value in Layer's weights array in CPU memory
	void fetchLayerWeights_Bias(Layer Lyr);

    //Free CPU memory used by Layer's data array
    void freeLayerNeuronData(Layer Lyr);

    //Free CPU memory used by Layer's weights array
    void freeLayerWeights_Bias(Layer Lyr);

    //Calculate error of each neurons in a layer and store error in Blue color
	void calcError(Layer Lyr, float* ActualOutput);

    //Train and update weights of the specified Layer
	void trainLayer(Layer Lyr, float* LearningRate);

    //Get errors from next layer neurons and backpropogate with weights to current layer neurons
	void backPropogateError(Layer Lyr);
    
    

    ////////////////////////////////////////////// Shader Codes /////////////////////////////////////////////////////

    const char* WeightInitShader_code =
        "#version 420                                                                    \n"
        "precision highp float;                                                          \n"   
        "precision highp image2D;                                                        \n"     
        "#extension GL_ARB_compute_shader : require                                      \n"
        "#extension GL_ARB_shader_image_load_store : require                             \n"
        "#extension GL_ARB_gpu_shader5 : require                                         \n"
        
        "#extension GL_ARB_gpu_shader_fp64 : require  \n"        
        "#pragma optionNV(fastmath off) \n"
        "#pragma optionNV(fastprecision off)  \n"
        "#pragma optionNV(strict on)    \n "

        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;                \n"
        "layout(rgba32f, binding = 0) uniform image2D img_output;                        \n"
        "uniform int seed;                                                               \n"

        "float rand(vec2 co)                                                             \n"
        "{                                                                               \n"
        // "   return (sin(dot(co, vec2(12.9898 , 78.233))) * 43758.5453) ;                 \n"                
        "   return fract(sin(dot(co, vec2(12.9898 , 78.233)))) ;                         \n"                
        "}                                                                               \n"
        "void main()                                                                     \n"
        "{                                                                               \n"
        "    vec4 CLR = vec4(rand(vec2( gl_GlobalInvocationID.x, tan(seed))), 0, 0, 1);  \n"
        "    if(CLR.r == 0) CLR.r = 0.001;                                               \n"        
        "    imageStore(img_output, ivec2(gl_GlobalInvocationID.xy ), CLR);              \n"
        "}                                                                               \0"
        ;

    const char* ActivationShader_code = 
        "#version 420                                                                    \n"
        "precision highp float;                                                          \n"
        "precision highp image2D;                                                        \n"
        "#extension GL_ARB_compute_shader : require                                      \n"
        "#extension GL_ARB_shader_image_load_store : require                             \n"
        "#extension GL_ARB_gpu_shader5 : require                                         \n"

        "#extension GL_ARB_gpu_shader_fp64 : require  \n"        
        "#pragma optionNV(fastmath off) \n"
        "#pragma optionNV(fastprecision off)  \n"
        "#pragma optionNV(strict on)    \n "

        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;                \n"
        "layout(rgba32f, binding = 0) uniform image2D img_output;                        \n"
        "layout(rgba32f, binding = 1) readonly uniform image2D PreviousLayer;            \n"
        "layout(rgba32f, binding = 2) readonly uniform image2D LayerWeight;              \n"
        "uniform int PreviousLayer_size;                                                 \n"
        "uniform int Weight_size;                                                        \n"
        "float weight_start;                                                             \n"
        "float bias_loc;                                                                 \n"
        "float val, Rval;                                                                \n"
        "vec4 fetch;                                                                     \n"
    
        "float Activate(float x);                                                        \n"

        "void main()                                                                     \n"
        "{                                                                               \n"
        "   weight_start = int(gl_GlobalInvocationID.x) * (PreviousLayer_size + 1);      \n"
        "   bias_loc = weight_start + PreviousLayer_size;                                \n"
        "   val = 0;                                                                     \n"
        "   Rval = 0;                                                                    \n"
        "   for(float i=0; i<PreviousLayer_size;i++)                                     \n"
        "   {                                                                            \n"
        "       fetch = imageLoad(PreviousLayer, ivec2(i,0));                            \n"
        "       val = fetch.r;                                                           \n"
        "       fetch = imageLoad(LayerWeight, ivec2(i+weight_start,0));                 \n"
        "       val *= fetch.r;                                                          \n"
        "       Rval += val;                                                             \n"
        "   }                                                                            \n"
        "   fetch = imageLoad(LayerWeight, ivec2(bias_loc,0));                           \n"
        "   Rval += fetch.r;                                                             \n"
        "   Rval = Activate(Rval);                                                       \n"
        "   imageStore(img_output, ivec2(gl_GlobalInvocationID.xy ), vec4(Rval,0,0,1));  \n"
        "}                                                                               \0"        
        ;

    const char* ErrorGen_code =     
        "#version 420                                                                 \n"
        "precision highp float;                                                       \n"
        "#extension GL_ARB_compute_shader : require                                   \n"
        "#extension GL_ARB_shader_image_load_store : require                          \n"
        "#extension GL_ARB_gpu_shader5 : require                                      \n"
        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;             \n"
        "layout(rgba32f, binding = 0) uniform image2D NeuronsOutput;                  \n"  
        "layout(rgba32f, binding = 1) uniform image2D ActualOutput;                   \n"        
        "vec4 neuronOut, aOut;                                                        \n"
        "float Derivate(float val);                                                   \n"		
        "void main()                                                                  \n"
        "{                                                                            \n"
        "   neuronOut = imageLoad(NeuronsOutput, ivec2(gl_GlobalInvocationID.xy));    \n"        
        "   aOut = imageLoad(ActualOutput, ivec2(gl_GlobalInvocationID.xy));          \n"   
        "   neuronOut.b = aOut.r - neuronOut.r;                                       \n"     
        "   neuronOut.b *= Derivate(neuronOut.r);                                     \n"        
        "   imageStore(NeuronsOutput, ivec2(gl_GlobalInvocationID.xy), neuronOut);    \n"
        "}                                                                            \0"
        ;
    
    const char* WeightUpdateShader_code = 
        "#version 420                                                                           \n"
        "precision highp float;                                                                 \n"
        "#extension GL_ARB_compute_shader : require                                             \n"
        "#extension GL_ARB_shader_image_load_store : require                                    \n"
        "#extension GL_ARB_gpu_shader5 : require                                                \n"
        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;                       \n"
        "layout(rgba32f, binding = 0) uniform image2D Weights;                                  \n"  
        "layout(rgba32f, binding = 1) uniform image2D NeuronsOutput;                            \n"          
        "layout(rgba32f, binding = 2) uniform image2D PreviousLayer;                            \n"        
        "uniform int PreviousLayer_size;                                                        \n"
        "uniform int NextLayer_size;                                                            \n"
        "uniform float LearningRate = 1;                                                        \n"
        "vec4 outputVal, weight, inputVal;                                                      \n"
        "float NeuronSelect, OutSelect;                                                         \n"        
        "void main()                                                                            \n"
        "{                                                                                      \n"
            //get weight
        "   weight = imageLoad(Weights, ivec2(gl_GlobalInvocationID.xy));                       \n"
            //figure out which output neuron this weight connects to
        // "   OutSelect = mod(gl_GlobalInvocationID.x, NextLayer_size );"
        // "   OutSelect = gl_GlobalInvocationID.x / (NextLayer_size+1);"
        "   OutSelect = gl_GlobalInvocationID.x / (PreviousLayer_size+1);"
            //get that output neuron value  
        "   outputVal = imageLoad(NeuronsOutput, ivec2(OutSelect,0));                           \n"
            //figure out which input neuron this weight belongs to
        "   NeuronSelect = mod(gl_GlobalInvocationID.x, PreviousLayer_size + 1);                \n"
            //get that input neuron value
        "   inputVal = imageLoad(PreviousLayer, ivec2(NeuronSelect / PreviousLayer_size, 0));   \n"
            //set bias to 1 (can also use == instead of >=)
        "   if(NeuronSelect >= PreviousLayer_size)                                              \n"
        "       inputVal.r = 1.0;                                                               \n"        
            //update weight        
        "   weight.r += outputVal.b * inputVal.r * LearningRate;                                \n"                
        "   imageStore(Weights, ivec2(gl_GlobalInvocationID.xy), weight);                       \n"
        "}                                                                                      \0"
        ;    

    
    const char* ErrorBackPropogate_code = 
        "#version 420                                                                                                   \n"
        "precision highp float;                                                                                         \n"
        "#extension GL_ARB_compute_shader : require                                                                     \n"
        "#extension GL_ARB_shader_image_load_store : require                                                            \n"
        "#extension GL_ARB_gpu_shader5 : require                                                                        \n"
        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;                                               \n"
        "layout(rgba32f, binding = 0) uniform image2D NeuronsOutput;                                                    \n"  
        "layout(rgba32f, binding = 1) uniform image2D NextLayerOutput;                                                  \n"          
        "layout(rgba32f, binding = 2) uniform image2D WeightsToNextLayer;                                               \n"        
        "uniform int OutputLayer_size;                                                                                  \n"
        "uniform int NextLayer_size;                                                                                    \n"
        "vec4 nextLayerNeuron, weight, neuron;                                                                          \n"
        "float ERROR = 0;                                                                                               \n"             
        "float Derivate(float val);                                                                                     \n"		
        "void main()                                                                                                    \n"
        "{                                                                                                              \n"        
        "   int weight_size = (OutputLayer_size +1) * NextLayer_size;                                                   \n"
        "   neuron = imageLoad(NeuronsOutput, ivec2(gl_GlobalInvocationID.xy));                                         \n"
        "   for(int i=0; i<NextLayer_size; i++)                                                                       \n"
        "   {                                                                                                           \n"
        "       nextLayerNeuron = imageLoad(NextLayerOutput, ivec2(i,0));                                               \n"
        "       weight = imageLoad(WeightsToNextLayer, ivec2(i*(OutputLayer_size+1) + int(gl_GlobalInvocationID.x),0)); \n"
        //////////// "imageStore(WeightsToNextLayer, ivec2(i*(OutputLayer_size+1) + int(gl_GlobalInvocationID.x),0), vec4(1));"
        // below are 3 different ways to backpropogate. 
        //    1 is not good but needs to be tested. 
        //    2 and 3 are same but in 3, sigmoid_derivation is multiplied to ERROR after the error summation.
        //    needs to find out which one to use 2 or 3 if output is going to more than one neuron.
        //    2022.5.16 : method 2 might be correct

//   /*1*/ "       ERROR += (nextLayerNeuron.b * weight.r) * Derivate(nextLayerNeuron.r);                                  \n"
 /*2*/ "       ERROR += Derivate(neuron.r) * (nextLayerNeuron.b * weight.r);                                            \n"
//   /*3*/ "       ERROR += (nextLayerNeuron.b * weight.r);                                                                \n"
        "   }                                                                                                           \n"
//   /*3*/ "       ERROR *= Derivate(neuron.r);                                                                            \n"
        "   neuron.b = ERROR;                                                                                           \n"
        "   imageStore(NeuronsOutput, ivec2(gl_GlobalInvocationID.xy), neuron);                                         \n"
        "}                                                                                                              \0"
        ;    

    const char* ActiveDeriveLibs_code = 
        "#version 420                                                                    \n"
        "precision highp float;                                                          \n"
        "#extension GL_ARB_compute_shader : require                                      \n"
        "#extension GL_ARB_shader_image_load_store : require                             \n"
        "#extension GL_ARB_gpu_shader5 : require                                         \n"
        "#extension GL_ARB_explicit_uniform_location : enable                            \n"
        
        "layout(location = 0) uniform int selection = 0;                                 \n"
        
        "float sigmoid(float x)                                                          \n"
        "{                                                                               \n"
        "   return 1.0/(1.0 + exp(-x));                                                  \n"
        "}                                                                               \n"
        "float sigmoid_derivative(float x)                                               \n"
        "{                                                                               \n"
        "       return x * (1.0 - x);                                                    \n"
        "}                                                                               \n"
        "float tanH(float x)                                                             \n"
        "{                                                                               \n"
        "       return 2/(1+exp(-2*x)) - 1;                                              \n"
        "}                                                                               \n"
        "float tanH_derivative(float x)                                                  \n"
        "{                                                                               \n"
        "       return 1 - x*x;                                                          \n"
        "}                                                                               \n"
        "float reLu(float x)                                                             \n"
        "{                                                                               \n"
        "       return max(0,x);                                                         \n"
        "}                                                                               \n"
        "float reLu_derivative(float x)                                                  \n"
        "{                                                                               \n"
        "       return step(x,0);                                                        \n"
        "}                                                                               \n"
        "float Activate(float x)                                                         \n"
        "{                                                                               \n"
        "   if(selection == 0)                                                           \n"
        "       return sigmoid(x);                                                       \n"
        "   else if(selection == 1)                                                      \n"
        "       return tanH(x);                                                          \n"
        "   else if(selection == 2)                                                      \n"
        "       return reLu(x);                                                          \n"
        "}                                                                               \n"            
        "float Derivate(float x)                                                         \n"
        "{                                                                               \n"
        "   if(selection == 0)                                                           \n"
        "       return sigmoid_derivative(x);                                            \n"
        "   else if(selection == 1)                                                      \n"
        "       return tanH_derivative(x);                                               \n"
        "   else if(selection == 2)                                                      \n"
        "       return reLu_derivative(x);                                               \n"
        "}                                                                               \0"
        ;
};


/*
 *********************************************************
   Below are all API vars and function for HermesNetwork
 *********************************************************
*/

//Bypass the namespace to make structure type public
typedef HermesNetwork::NeuralNetwork NeuralNetwork;

//This enum stores IDs of all different Activation Functions. To be used as an argument in SetActivation()
enum ActivationType 
{   Sigmoid = 0, 
    TanH = 1, 
    ReLu = 2,
    LeakyReLu = 3
};

//Setup gl context, compile shaders, create drawing polygon
bool InitNeuralLink(bool GL_Context_Shared);

//Builds network with given input size, hiddenlayer size as array and output size.
template <typename T = int>
NeuralNetwork NetworkBuilder(int InputSize, std::initializer_list<T> HiddenLayers, int OutputSize);

//Activates every neurons of a layer at specified depth
void TriggerLayer(NeuralNetwork* Network, int LayerDepth);

//Triggers every layers in network sequentially from input layer to output layer
void TriggerNetwork(NeuralNetwork* Network);

//Set every neurons of input layer with specified values in array
void SendInputs(NeuralNetwork Network, float Inputs[]);

//Get neurons data in output layer as array
void FetchOutputLayerData(NeuralNetwork Network);

//Generate Error in output neurons, backpropogate errors to previous layers and updates every weight and bias
void TrainNetwork(NeuralNetwork Network, float ActualOutput[], float LearningRate);

//save network structure,weights and bias in a file.
void SaveNetwork(NeuralNetwork Network, const char filename[]);

//load saved network from disk and generate a live neural network as per saved data such as weights, bias and no of layers.
NeuralNetwork LoadNetwork(const char filename[]);

//Set Activation Function for all the layers in NeuralNetwork
void SetActivation(NeuralNetwork Network, ActivationType AllLayersType);

//Set different Activation Function for hidden and output layers in NeuralNetwor
void SetActivation(NeuralNetwork Network, ActivationType HiddenLayersType, ActivationType OutputLayersType);

//This function does nothing except changing network layer texture representation to a smooth, linear gradient.
void Terrify(NeuralNetwork N);

//This function does nothing except changing network layer texture representation rigid, fixed bars.
void DeTerrify(NeuralNetwork N);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////






//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////  DEFINITIONS  ////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////// HermesNetwork's functions ////////////////////////////////////////////////////////
HermesNetwork::Layer HermesNetwork::initLayer(int size, layerType typ)
{        
    Layer newL = new LayerHandle();
	newL->type = typ;
	newL->no_neuron = size;
    
    glGenTextures(1, &newL->NeuronsTex);
    glBindTexture(GL_TEXTURE_2D, newL->NeuronsTex);    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, newL->no_neuron, 1, 0, GL_RGBA, GL_FLOAT, NULL);	
    glBindTexture(GL_TEXTURE_2D, 0);

    return newL;
}

NeuralNetwork HermesNetwork::createBasicNetwork(int InputSize, int OutputSize)
{
    /* build input Layer structure */
    Layer inp = initLayer(InputSize, inputL);

    /* build output Layer structure */
    Layer op = initLayer(OutputSize, inputL);

    /* build NeuralNetowkr structure */
    NeuralNetwork nn = new NeuralNetworkHandle();
	nn->no_of_input = InputSize;
	nn->no_of_output = OutputSize;
	nn->inputLayer = inp;
	nn->outputLayer = op;      

    nn->inputLayer->next = op;
    nn->outputLayer->prev = nn->inputLayer;      

    return nn;
}

void HermesNetwork::appendHiddenLayer(NeuralNetwork Network, int LayerSize)
{
    Layer newL = initLayer(LayerSize, hiddenL);
    Network->outputLayer->prev->next = newL;
    newL->prev = Network->outputLayer->prev;
    newL->next = Network->outputLayer;
    Network->outputLayer->prev = newL;

    Network->no_layers++;
}

void HermesNetwork::connectLayer(Layer prev, Layer next)
{
	/*TODO
	* check and delete next->WeightsTex and generate new one as per previous added layer no. of neurons
	* vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	*/	
    if (next->WeightsTex > 0)  //delete old texture RBO if next layer already connected to an old previous layer
	{
		glDeleteTextures(1, &next->WeightsTex);
	}
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    /* init next's weight texture */
	next->no_weight = (prev->no_neuron + 1) * next->no_neuron;
	glGenTextures(1, &next->WeightsTex);
	glBindTexture(GL_TEXTURE_2D, next->WeightsTex);	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);	
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, next->no_weight, 1, 0, GL_RGBA, GL_FLOAT, NULL);	

    glBindImageTexture(0, next->WeightsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);    
    
    glUseProgram(WeightInit);

    glUniform1i(WINT_unifm_seed, rand());    
    glDispatchCompute(next->no_weight,1,1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

void HermesNetwork::triggerLayer(Layer Lyr)
{	
    glBindImageTexture(0, Lyr->NeuronsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);    
    glBindImageTexture(ACTV_unifm_prev_L_TEX, Lyr->prev->NeuronsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);    
    glBindImageTexture(ACTV_unifm_Layer_weight, Lyr->WeightsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);    
    
    glUseProgram(Activation);

    glUniform1i(ACTVLibs_unifm_SEL, Lyr->AFun);
    glUniform1i(ACTV_unifm_prev_size, Lyr->prev->no_neuron);
    glUniform1i(ACTV_unifm_weight_size, Lyr->no_weight);
    glDispatchCompute(Lyr->no_neuron,1,1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);    
}

void fetchLayerNeuronsData_ERR(HermesNetwork::Layer Lyr)
{
    using namespace HermesNetwork;
    if(!Lyr->data)
        Lyr->data = new float[Lyr->no_neuron];
    glBindTexture(GL_TEXTURE_2D, Lyr->NeuronsTex);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_BLUE, GL_FLOAT, Lyr->data);
    glBindTexture(GL_TEXTURE_2D, 0);    
}

void HermesNetwork::fetchLayerNeuronsData(Layer Lyr)
{
    if(!Lyr->data)
        Lyr->data = new float[Lyr->no_neuron];
    glBindTexture(GL_TEXTURE_2D, Lyr->NeuronsTex);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, Lyr->data);
    glBindTexture(GL_TEXTURE_2D, 0);    
}

void HermesNetwork::fetchLayerWeights_Bias(Layer Lyr)
{
    if(!Lyr->weights)
        Lyr->weights = new float[Lyr->no_weight];
    glBindTexture(GL_TEXTURE_2D, Lyr->WeightsTex);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, Lyr->weights);
    glBindTexture(GL_TEXTURE_2D, 0);    
}

void HermesNetwork::freeLayerNeuronData(Layer Lyr)
{
    if(Lyr->type == hiddenL)
    {
        delete[] Lyr->data;        
        Lyr->data = nullptr;
    }
}
    
void HermesNetwork::freeLayerWeights_Bias(Layer Lyr)
{
    if(Lyr->type != inputL)
    {
        delete[] Lyr->weights;
        Lyr->weights = nullptr;
    }
}

void HermesNetwork::calcError(Layer Lyr, float* ActualOutput)
{
    /* convert output array to texture */
	glBindTexture(GL_TEXTURE_2D, TempTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, Lyr->no_neuron, 1, 0, GL_RED, GL_FLOAT, ActualOutput);
    
    glBindImageTexture(ERROR_unifm_neuronOut_TEX, Lyr->NeuronsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);    
    glBindImageTexture(ERROR_unifm_actualOut_TEX, TempTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);    

    glUseProgram(ErrorGen);

    glUniform1i(ACTVLibs_unifm_SEL, Lyr->AFun);
    glDispatchCompute(Lyr->no_neuron,1,1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);  

    glBindTexture(GL_TEXTURE_2D, 0);
}

void HermesNetwork::trainLayer(Layer Lyr, float* LearningRate)
{
    glBindImageTexture(WGHTUP_unifm_weight_TEX, Lyr->WeightsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);    
    glBindImageTexture(WGHTUP_unifm_neuronOut_TEX, Lyr->NeuronsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);    
    glBindImageTexture(WGHTUP_unifm_prev_L_TEX, Lyr->prev->NeuronsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);    
    
    glUseProgram(WeightUpdate);
    
    glUniform1i(WGHTUP_unifm_prev_size, Lyr->prev->no_neuron);
    glUniform1i(WGHTUP_unifm_next_size, Lyr->no_neuron);
    glUniform1i(WGHTUP_unifm_LearnRT, *LearningRate);
    glDispatchCompute(Lyr->no_weight,1,1);    
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);    
}

void HermesNetwork::backPropogateError(Layer Lyr)
{
    glBindImageTexture(ERROR_BP_unifm_neuronOut_TEX, Lyr->NeuronsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);    
    glBindImageTexture(ERROR_BP_unifm_next_L_TEX, Lyr->next->NeuronsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);    
    glBindImageTexture(ERROR_BP_unifm_weight_TEX, Lyr->next->WeightsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);    
    
    glUseProgram(ErrorBackPropogate);

    glUniform1i(ACTVLibs_unifm_SEL, Lyr->AFun);
    glUniform1i(ERROR_BP_unifm_Layer_size, Lyr->no_neuron);
    glUniform1i(ERROR_BP_unifm_next_L_size, Lyr->next->no_neuron);
    glDispatchCompute(Lyr->no_neuron,1,1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);    
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////// public functions  /////////////////////////////////////////////////////////////////
bool InitNeuralLink(bool GL_Context_Shared = false)
{
    using namespace HermesNetwork;    
    if(!GL_Context_Shared)
    {
        //Create window context
        #ifdef _WIN32
                HWND offscreen_context = ::CreateWindowA("STATIC", "OpenGL Context Space", 0 , 0, 0, 0, 0, NULL, NULL, NULL, NULL);
                PIXELFORMATDESCRIPTOR pfd =
                        {
                                sizeof(PIXELFORMATDESCRIPTOR),1,
                                PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER, PFD_TYPE_RGBA,
                                32,0,0,0,0,0,0,
                                0,0,0,0,0,0,
                                0,24,8,0,
                                PFD_MAIN_PLANE,0,0, 0, 0
                        };
                HDC DC = GetDC(offscreen_context);
                int  letWindowsChooseThisPixelFormat;
                letWindowsChooseThisPixelFormat = ChoosePixelFormat(DC, &pfd);
                SetPixelFormat(DC,letWindowsChooseThisPixelFormat, &pfd);
                GlRenderingContext = wglCreateContext(DC);
                wglMakeCurrent (DC, GlRenderingContext);
                //wglDeleteContext(GLRenderingContext);
        #endif
        #ifdef __linux__
                //call linux api to create invisible window
                            //attatch gl viewport
                            //make gl context
        #endif
        #ifdef __APPLE__
                //call macOS api to create invisible window
                            //attatch gl viewport
                            //make gl context
        #endif
        #ifdef __ANDROID__
                //call macOS api to create invisible window
                            //attatch gl viewport
                            //make gl context
        #endif
    }


    //Init openGl context
    #ifdef __glew_h__
        glewInit();
    #endif
    #ifdef __glut_h__
        glutInit(NULL,NULL);
    #endif
    #ifdef  __FREEGLUT_H__
        glutInit(NULL,NULL);
    #endif

    /* Create And Compile Shaders */
    int WeightInitComputetShader = glCreateShader(GL_COMPUTE_SHADER);
    int ActivationComputeShader = glCreateShader(GL_COMPUTE_SHADER);
    int ErrorGenComputeShader = glCreateShader(GL_COMPUTE_SHADER);
    int WeightUpdateComputeShader = glCreateShader(GL_COMPUTE_SHADER);   
    int ErrorBPComputeShader = glCreateShader(GL_COMPUTE_SHADER);
    int Active_DeriveLibsComputeShader = glCreateShader(GL_COMPUTE_SHADER);
           

    glShaderSource(WeightInitComputetShader, 1, &WeightInitShader_code, NULL);
    glShaderSource(ActivationComputeShader, 1, &ActivationShader_code, NULL);
    glShaderSource(ErrorGenComputeShader, 1, &ErrorGen_code, NULL);
    glShaderSource(WeightUpdateComputeShader, 1, &WeightUpdateShader_code, NULL);
    glShaderSource(ErrorBPComputeShader, 1, &ErrorBackPropogate_code, NULL);
    glShaderSource(Active_DeriveLibsComputeShader, 1, &ActiveDeriveLibs_code, NULL);



    glCompileShader(WeightInitComputetShader);
    glCompileShader(ActivationComputeShader);
    glCompileShader(ErrorGenComputeShader);
    glCompileShader(WeightUpdateComputeShader);
    glCompileShader(ErrorBPComputeShader);
    glCompileShader(Active_DeriveLibsComputeShader);
   

    

    /* handle compile error	*/
    int success;
    bool fail;
    glGetShaderiv(WeightInitComputetShader, GL_COMPILE_STATUS, &success); fail |= !success;
    std::cout<<success<<fail<<"\n";
    glGetShaderiv(ActivationComputeShader, GL_COMPILE_STATUS, &success); fail |= !success;
    std::cout<<success<<fail<<"\n";
    glGetShaderiv(ErrorGenComputeShader, GL_COMPILE_STATUS, &success); fail |= !success;
    std::cout<<success<<fail<<"\n";
    glGetShaderiv(WeightUpdateComputeShader, GL_COMPILE_STATUS, &success); fail |= !success;
    std::cout<<success<<fail<<"\n";
    glGetShaderiv(ErrorBPComputeShader, GL_COMPILE_STATUS, &success); fail |= !success;
    std::cout<<success<<fail<<"\n";
    glGetShaderiv(Active_DeriveLibsComputeShader, GL_COMPILE_STATUS, &success); fail |= !success;
    std::cout<<success<<fail<<"\n";


    char infoLog[512];
    glGetShaderInfoLog(Active_DeriveLibsComputeShader, 512, NULL, infoLog);
    std::cout<<infoLog;
    if(fail)
        return !fail;



    /* remove shader source codes from memory	*/


    /* bind shader programs */
    WeightInit = glCreateProgram();
    glAttachShader(WeightInit, WeightInitComputetShader);
    glLinkProgram(WeightInit);

    Activation = glCreateProgram();
    glAttachShader(Activation, Active_DeriveLibsComputeShader);
    glAttachShader(Activation, ActivationComputeShader);
	glLinkProgram(Activation);

    ErrorGen = glCreateProgram();
    glAttachShader(ErrorGen, Active_DeriveLibsComputeShader);
    glAttachShader(ErrorGen, ErrorGenComputeShader);
	glLinkProgram(ErrorGen);

    WeightUpdate = glCreateProgram();
    glAttachShader(WeightUpdate, WeightUpdateComputeShader);
	glLinkProgram(WeightUpdate);

    ErrorBackPropogate = glCreateProgram();
    glAttachShader(ErrorBackPropogate, Active_DeriveLibsComputeShader);
    glAttachShader(ErrorBackPropogate, ErrorBPComputeShader);
	glLinkProgram(ErrorBackPropogate);

    /* remove compiled shader from RAM */
    glDeleteShader(WeightInitComputetShader);
    glDeleteShader(ActivationComputeShader);
    glDeleteShader(ErrorGenComputeShader);
    glDeleteShader(WeightUpdateComputeShader);   
    glDeleteShader(ErrorBPComputeShader);    
    glDeleteShader(Active_DeriveLibsComputeShader);    
   

    /* Initialize Temporary Texture for misc operations */
    glGenTextures(1, &TempTex);
    glBindTexture(GL_TEXTURE_2D, TempTex);
    /*glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, newL->no_neuron, 1, 0, GL_RGB, GL_FLOAT, NULL);*/
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);


    /* Get all uniform locations */

    //explicit function selector location as defined in ActiveDeriveLibs shader
    //explicit location used to maintain common location across all sharing compute shader program
    //take care that no other uniforms in ohter shader is explicitly in same location
    ACTVLibs_unifm_SEL = 0;


    glUseProgram(WeightInit);
    WINT_unifm_seed = glGetUniformLocation(WeightInit, "seed");

    glUseProgram(Activation);
	ACTV_unifm_prev_size = glGetUniformLocation(Activation, "PreviousLayer_size");
	ACTV_unifm_weight_size = glGetUniformLocation(Activation, "Weight_size");
	ACTV_unifm_prev_L_TEX = 1;    // from shader uniform layout binding
	ACTV_unifm_Layer_weight = 2;  // from shader uniform layout binding  
    ACTVLibs_unifm_SEL = glGetUniformLocation(Activation, "selection");

    std::cout<<"LOC: "<< glGetUniformLocation(Activation, "selection") << " "<<  ACTV_unifm_prev_size ;

    glUseProgram(ErrorGen);
    ERROR_unifm_neuronOut_TEX = 0; // from shader uniform layout binding
    ERROR_unifm_actualOut_TEX = 1; // from shader uniform layout binding
    std::cout<<"LOC: "<< glGetUniformLocation(Activation, "selection") << " "<<  glGetUniformLocation(ErrorGen, "NeuronsOutput") ;

    glUseProgram(WeightUpdate);
    WGHTUP_unifm_next_size = glGetUniformLocation(WeightUpdate, "NextLayer_size");
    WGHTUP_unifm_prev_size = glGetUniformLocation(WeightUpdate, "PreviousLayer_size");
    WGHTUP_unifm_LearnRT = glGetUniformLocation(WeightUpdate, "LearningRate");
    WGHTUP_unifm_weight_TEX = 0;    // from shader uniform layout binding
    WGHTUP_unifm_neuronOut_TEX = 1; // from shader uniform layout binding
    WGHTUP_unifm_prev_L_TEX = 2;    // from shader uniform layout binding    

    glUseProgram(ErrorBackPropogate);
    ERROR_BP_unifm_neuronOut_TEX = 0; // from shader uniform layout binding
	ERROR_BP_unifm_next_L_TEX = 1;    // from shader uniform layout binding
	ERROR_BP_unifm_weight_TEX = 2;    // from shader uniform layout binding
	ERROR_BP_unifm_Layer_size = glGetUniformLocation(ErrorBackPropogate, "OutputLayer_size");
	ERROR_BP_unifm_next_L_size = glGetUniformLocation(ErrorBackPropogate, "NextLayer_size"); 
    std::cout<<"LOC: "<<  ERROR_BP_unifm_Layer_size << glGetUniformLocation(ErrorBackPropogate, "selection");   
       

    srand(time(0));
    return true;
}

template <typename T = int>
NeuralNetwork NetworkBuilder(int InputSize, std::initializer_list<T> HiddenLayers, int OutputSize)
{    
    using namespace HermesNetwork;
    NeuralNetwork nn = createBasicNetwork(InputSize, OutputSize);
    for(int s: HiddenLayers)
        appendHiddenLayer(nn,s);
        
    Layer l = nn->inputLayer;
    while(l->next != nullptr)
    {
        connectLayer(l, l->next);
        l = l->next;        
    }

    //permanently bind output layer with Out array
    fetchLayerNeuronsData(nn->outputLayer);
    nn->Out = nn->outputLayer->data;
    return nn;    
}

void TriggerLayer(NeuralNetwork Network, int LayerDepth)
{
	using namespace HermesNetwork;

    /* Don't allow input layer activation */
    if(LayerDepth == 0)
        return;

	/* Select Layer at given depth */
	Layer Lyr = Network->inputLayer;
	for (int i = 0; i < LayerDepth; i++)
		Lyr = Lyr->next;

	/* Trigger selected layer */
	triggerLayer(Lyr);
}

void TriggerNetwork(NeuralNetwork Network)
{
	using namespace HermesNetwork;

	Layer Lyr = Network->inputLayer;    
	for (int i = 1; i < Network->no_layers; i++)
	{
		Lyr = Lyr->next;
		triggerLayer(Lyr);
	}
}

void SendInputs(NeuralNetwork Network, float Inputs[])
{
	glBindTexture(GL_TEXTURE_2D, Network->inputLayer->NeuronsTex);		
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, Network->inputLayer->no_neuron, 1, 0, GL_RED, GL_FLOAT, Inputs);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void FetchOutputLayerData(NeuralNetwork Network)
{
    HermesNetwork::fetchLayerNeuronsData(Network->outputLayer);
}

void TrainNetwork(NeuralNetwork Network, float ActualOutput[], float LearningRate = 1.0)
{    
    using namespace HermesNetwork;
    {
        /*
        *      STEPS FOR BACKPORPAGATION (old)
        *
        *    1.  calculate error for output layer                    --  for each neuron: error = sigmoid_derivative(neuron_output * (ActualOutput-neuron_output))
        *    2.  adjust weights for output layer according to error  --  for each weights coming from neuron m, going to a neuron n: weights += n.error*m.value
        *    3.  calculate error for hidden layer                    --  same as that of output layer error calculation
        *    4.  adjust weights for hidden layer according to error  --  same as that of output layer weight adjustment
        */

        // /*1*/ calcError(Network->outputLayer, ActualOutput);
        // /*2*/ trainLayer(Network->outputLayer, &LearningRate);
        
        // Layer Lyr = Network->outputLayer->prev;
        // for(int i = Network->no_layers; i > 2; i--)
        // {
        //     /*3*/ backPropogateError(Lyr);
        //     /*4*/ trainLayer(Lyr, &LearningRate);
        //     Lyr = Lyr->prev;
        // }
    }
    /*
     *      STEPS FOR BACKPORPAGATION (new)
     *
     *    1.  calculate error for output layer                    --  for each neuron: error = sigmoid_derivative(neuron_output * (ActualOutput-neuron_output))
     *    3.  calculate error for hidden layer                    --  same as that of output layer error calculation
     *    2.  adjust weights for output layer according to error  --  for each weights coming from neuron m, going to a neuron n: weights += n.error*m.value    
     *    4.  adjust weights for hidden layer according to error  --  same as that of output layer weight adjustment
     */
    
    /*1*/ calcError(Network->outputLayer, ActualOutput);
    // /*2*/ trainLayer(Network->outputLayer, &LearningRate);
    
    Layer Lyr = Network->outputLayer->prev;
	for(int i = Network->no_layers; i > 2; i--)
    {
        /*3*/ backPropogateError(Lyr);
        // /*4*/ trainLayer(Lyr, &LearningRate);
        Lyr = Lyr->prev;
    }

 /*2*/ trainLayer(Network->outputLayer, &LearningRate);
    Lyr = Network->outputLayer->prev;
	for(int i = Network->no_layers; i > 2; i--)
    {
        // /*3*/ backPropogateError(Lyr);
        /*4*/ trainLayer(Lyr, &LearningRate);
        Lyr = Lyr->prev;
    }
    
}

void SaveNetwork(NeuralNetwork Network, const char filename[])
{
    /*
     *      FILE STRUCTURE
     *
     * -------------------------------------------
     *  No of Layers | Input Size | Output Size             -int,int,int
     * -------------------------------------------
     *  1st Hidden Layer Size | [Array of weights]          -int,[float]
     * -------------------------------------------
     *  2nd Hidden Layer Size | [Array of weights]          -int,[float]
     * -------------------------------------------
     *      :
     *      :
     *  nth Hidden Layer Size | [Array of weights]          -int,[float]
     * -------------------------------------------
     *  [Array of weights of Output Layer]                  -[float]
     * -------------------------------------------
     */
    
    std::fstream file;
    file.open(filename,std::ios::out|std::ios::binary);
    file.write((char*)&Network->no_layers, sizeof(int));
    file.write((char*)&Network->no_of_input, sizeof(int));
    file.write((char*)&Network->no_of_output, sizeof(int));

    HermesNetwork::Layer L = Network->inputLayer->next;
    for(int i=1; i < Network->no_layers-1 ; i++,L = L->next)
    {
        file.write((char*)&L->no_neuron, sizeof(int));
        fetchLayerWeights_Bias(L);
        file.write((char*)L->weights, sizeof(float) * L->no_weight);
        freeLayerWeights_Bias(L);
    }

    fetchLayerWeights_Bias(Network->outputLayer);    
    file.write((char*)Network->outputLayer->weights, sizeof(float) * Network->outputLayer->no_weight);    
    freeLayerWeights_Bias(Network->outputLayer);
    file.close();
}

NeuralNetwork LoadNetwork(const char filename[])
{
    /*
     *      FILE STRUCTURE
     *
     * -------------------------------------------
     *  No of Layers | Input Size | Output Size             -int,int,int
     * -------------------------------------------
     *  1st Hidden Layer Size | [Array of weights]          -int,[float]
     * -------------------------------------------
     *  2nd Hidden Layer Size | [Array of weights]          -int,[float]
     * -------------------------------------------
     *      :
     *      :
     *  nth Hidden Layer Size | [Array of weights]          -int,[float]
     * -------------------------------------------
     *  [Array of weights of Output Layer]                  -[float]
     * -------------------------------------------
     */

    std::fstream file;
    file.open(filename, std::ios::in| std::ios::binary);
    if(!file.is_open())
        return nullptr;
    int layerSize, inputSize, outputSize;
    file.read((char*)&layerSize,sizeof(int));
    file.read((char*)&inputSize,sizeof(int));
    file.read((char*)&outputSize,sizeof(int));
    

    NeuralNetwork Network = HermesNetwork::createBasicNetwork(inputSize, outputSize);    
    HermesNetwork::Layer L = Network->inputLayer;
    int hlSize = 0;    

    for(int i = 1; i < layerSize-1; i++)
    {
        file.read((char*)&hlSize,sizeof(int));
        std::cout<<"\nhlS: "<<hlSize;        
        HermesNetwork::appendHiddenLayer(Network,hlSize);
        HermesNetwork::connectLayer(L,L->next);     

        L = L->next;        
        float *hlWeights = new float[L->no_weight];
        file.read((char*)hlWeights, sizeof(float) * L->no_weight);        
        glBindTexture(GL_TEXTURE_2D, L->WeightsTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, L->no_weight, 1, 0, GL_RED, GL_FLOAT, hlWeights);
        glBindTexture(GL_TEXTURE_2D, 0);
        delete[] hlWeights;        
    }

    HermesNetwork::connectLayer(L,L->next);     

    L = Network->outputLayer;
    float *outputWeights = new float[L->no_weight];
    file.read((char*)outputWeights, sizeof(float) * L->no_weight);    
    glBindTexture(GL_TEXTURE_2D, L->WeightsTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, L->no_weight, 1, 0, GL_RED, GL_FLOAT,outputWeights);    
    glBindTexture(GL_TEXTURE_2D, 0);
    delete[] outputWeights;


    //permanently bind output layer with Out array
    fetchLayerNeuronsData(Network->outputLayer);
    Network->Out = Network->outputLayer->data;    

    file.close();
    return Network;
}

void SetActivation(NeuralNetwork Network, ActivationType AllLayersType)
{
    using namespace HermesNetwork;

	Layer Lyr = Network->inputLayer;    
	for (int i = 1; i < Network->no_layers; i++)
	{
		Lyr = Lyr->next;
		Lyr->AFun = AllLayersType;
	}
    
}

void SetActivation(NeuralNetwork Network, ActivationType HiddenLayersType, ActivationType OutputLayersType)
{
    SetActivation(Network,HiddenLayersType);
    Network->outputLayer->AFun = OutputLayersType;
}

void Terrify(NeuralNetwork N)
{    
    HermesNetwork::Layer l = N->inputLayer;   
    while(l != nullptr)
        {
            glBindTexture(GL_TEXTURE_2D, l->WeightsTex);    
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0);

            glBindTexture(GL_TEXTURE_2D, l->NeuronsTex);    
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0);
            l = l->next;            
        }                    
}

void DeTerrify(NeuralNetwork N)
{    
    HermesNetwork::Layer l = N->inputLayer;   
    while(l != nullptr)
    {
        glBindTexture(GL_TEXTURE_2D, l->WeightsTex);    
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindTexture(GL_TEXTURE_2D, l->NeuronsTex);    
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
        l = l->next;            
    }                    
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////






//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif //__HERMES_NETWORK__