/*
 * #define HN_SharedGLcontext
 *      If the main program is also using a OpenGL context for anything other than NeuralNetwork processing,
 *      then defining this macro before "include<HermesNetwork.h>" statement in main program will make
 *      HermesNetwork share OpenGL graphics context with other graphics processing task like UI, 2D/3D rendering etc.
 *      But if this macro is defined, then before calling InitNeuralLink(), a GL context must be already created and initialized
 *
 *
*/

#ifndef __HERMES_NETWORK__
#define __HERMES_NETWORK__

#include<initializer_list>
#include<fstream>

#ifdef _WIN32
    #include<windows.h>
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


//Handle to entire network.
//[Always create a pointer object of it]
struct NeuralNetwork;

//Setup gl context, compile shaders, create drawing polygon
void InitNeuralLink();

//Builds network with given input size, hiddenlayer size as array and output size.
template <typename T = int>
NeuralNetwork* NetworkBuilder(int InputSize, std::initializer_list<T> HiddenLayers, int OutputSize);

//Adds a new layer at given depth. No depth or negative depth will add layer at last before output layer
void AddLayer(NeuralNetwork* Network, int size, unsigned int Depth = -1);

//Activates every neurons of a layer at specified depth
void TriggerLayer(NeuralNetwork* Network, int LayerDepth);

//Triggers every layers in network sequentially from input layer to output layer
void TriggerNetwork(NeuralNetwork* Network);

//Set every neurons of input layer with specified values in array
void SendInputs(NeuralNetwork* Network, float Inputs[]);

//Returns values of every neurons in output layer as array
float* GetOutputLayerData(NeuralNetwork* Network);

//Generate Error in output neurons, backpropogate errors to previous layers and updates every weight and bias
void TrainNetwork(NeuralNetwork* Network, float ActualOutput[], float LearningRate = 1.0);

//save network structure,weights and bias in a file.
void SaveNetwork(NeuralNetwork* Network, char filename[]);

//load saved network from disk and generate a live neural network as per saved data such as weights, bias and no of layers.
NeuralNetwork* LoadNetwork(char filename[]);

//HermesNetwork library builds neural network inside GPU and triggers network layers as per command.
//The Library uses OpenGL to access GPUs. It can be run on any dedicated or integrated GPU regardless of GPU vendor's drivers, OS or hardware architecture.
//It provides features to add layers at any depth, send input data to input layer, retrieve data or weights from any layer.
//Additional features to be implemented are saving network structure and its weights in a file and loading it.
namespace HermesNetwork
{
	//////////////////////////////////////////// Objects ///////////////////////////////////////
	enum layerType	 {	inputL, outputL, hiddenL };
	enum networkType {	convolutional };

	struct Layer
	{
		int no_neuron = NULL;
		int no_weight = NULL;
		layerType type;
		Layer* next = nullptr;
		Layer* prev = nullptr;
		unsigned int NeuronsFbo = NULL;
		unsigned int NeuronsTex = NULL;
		unsigned int WeightFbo = NULL;
		unsigned int WeightTex = NULL;
	};

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

	unsigned int VBO;
	unsigned int VAO;
	unsigned int EBO;
	unsigned int TempTex;
	int WeightInit;
	int SigmoidActivation;
	int WeightUpdate;
	int ErrorGen;
	int ErrorBackPropogate;

	GLint WINT_unifm_no_of_weight;
	GLint SIGACT_unifm_prev_size;
	GLint SIGACT_unifm_weight_size;
	GLint SIGACT_unifm_prev_L_TEX;
	GLint SIGACT_unifm_Layer_weight;

	GLint WGHTUP_unifm_prev_size;
	GLint WGHTUP_unifm_LearnRT;
	GLint WGHTUP_unifm_weight_TEX;
	GLint WGHTUP_unifm_prev_L_TEX;
	GLint WGHTUP_unifm_neuronOut_TEX;

	GLint ERROR_unifm_neuronOut_TEX;
	GLint ERROR_unifm_actualOut_TEX;

	GLint ERROR_BP_unifm_neuronOut_TEX;
	GLint ERROR_BP_unifm_next_L_TEX;
	GLint ERROR_BP_unifm_weight_TEX;
	GLint ERROR_BP_unifm_Layer_size;
	GLint ERROR_BP_unifm_next_L_size;

	///////////////////////////////////////////// Functions /////////////////////////////////////////////////////////

	//This function will be called whenever a new layer is created
	Layer* initLayer(int size, layerType typ);

	//Creates network with only input and output. Returns a pointer of type- NeuralNetwork
	NeuralNetwork* createNetwork(int InputSize, int OutputSize);

	//Connect two layer with weight and initialize weights
	void connectLayer(NeuralNetwork* Network, Layer* prev, Layer* next);

	//Returns an array of all the weights (along with bias after list of weight of a neuron) of layer at the specified depth
	float* getWeights_Bias(NeuralNetwork* Network, int LayerDepth);

	//Activates every neurons of the given layer
	void triggerLayer(Layer* Lyr);

	//Returns array of data of every neuron from layer at specified depth
	float* getLayerNeuronsData(NeuralNetwork* Network, int LayerDepth);

	//Train and update weights of the specified Layer
	void trainLayer(Layer* Lyr, float* LearningRate);

	//Calculate error of each neurons in a layer and store error in Blue color
	void calcError(Layer* Lyr, float* ActualOutput);

	//Get errors from next layer neurons and backpropogate with weights to current layer neurons
	void backPropogateError(Layer* Lyr);

	////////////////////////////////////////////// Shader Codes ////////////////////////////////////////////////////
	const char* vertexShader_code =
		"#version 330 core                                              \n"
		"layout(location = 0) in vec3 aPos;                             \n"
		"layout(location = 1) in vec3 aColor;                           \n"
		"layout(location = 2) in vec2 aTexCoord;                        \n"
		"out vec2 TexCoord;                                             \n"
		"void main()                                                    \n"
		"{                                                              \n"
		"   gl_Position = vec4(aPos.x, -aPos.y, aPos.z  , 1.0);         \n"
		"   TexCoord = vec2(aTexCoord.x, aTexCoord.y);                  \n"
		"}                                                              \0"
		;

	const char* sigmoidActivationShader_code =
		"#version 330 core                                                                              \n"
		"precision highp float;                                                                         \n"
		"uniform sampler2D PreviousLayer;                                                               \n"
		"uniform sampler2D LayerWeight;                                                                 \n"
		"uniform int PreviousLayer_size;                                                                \n"
		"uniform int Weight_size;                                                                       \n"
		"in vec2 TexCoord;                                                                              \n"
		"in vec4 gl_FragCoord;                                                                          \n"
		"out vec4 FragColor;                                                                            \n"
		"float weight_start;                                                                            \n"
		"float bias_loc;                                                                                \n"
		"float val;                                                                                     \n"
		"vec4 fetch;                                                                                    \n"

		"float sigmoid(float x)                                                                         \n"
		"{                                                                                              \n"
		"       return 1.0/(1.0 + exp(-x));                                                             \n"
		"}                                                                                              \n"

		"float getCoord(float index, int size)                                                          \n"
		"{                                                                                              \n"
		        //"return ( index/size + (index+1)/size )/2.0 ;\n;" below is same formula but optimized by evaluating
		"       return (2*index +1)/ (2*size);                                                          \n"
		"}                                                                                              \n"

		"void main()                                                                                    \n"
		"{                                                                                              \n"
		"       weight_start = int(gl_FragCoord.x) * (PreviousLayer_size + 1);                          \n"
		"       bias_loc = weight_start + PreviousLayer_size;                                           \n"
		"       fetch;                                                                                  \n"
		"       val = 0;                                                                                \n"
		"       for(float i=0; i<PreviousLayer_size;i++)                                                \n"
		"       {                                                                                       \n"
		"           fetch = texture(PreviousLayer, vec2(getCoord(i,PreviousLayer_size),0));             \n"
		"           val = fetch.r;                                                                      \n"
		"           fetch = texture(LayerWeight, vec2(getCoord(i+weight_start,Weight_size),0));         \n"
		"           val *= fetch.r;                                                                     \n"
		"           FragColor.r += val;                                                                 \n"
		"       }                                                                                       \n"
		"       fetch = texture(LayerWeight, vec2(getCoord(bias_loc,Weight_size),0));                   \n"
		"       FragColor.r += fetch.r;                                                                 \n"
		"       FragColor.r = sigmoid(FragColor.r);                                                     \n"
		"}                                                                                              \0"
		;

	const char* WeightInitShader_code =
		"#version 330 core                                                                                                  \n"
		"precision highp float;                                                                                             \n"
		"out vec4 FragColor;                                                                                                \n"
		"in vec2 TexCoord;                                                                                                  \n"
		"uniform int no_weight;                                                                                             \n"
//		"uniform sampler2D Texture;                                                                                         \n"
		"in vec4 FragCoord;                                                                                                 \n"

		"float rand(vec2 co)                                                                                                \n"
		"{                                                                                                                  \n"
		"       return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);                                             \n"
		"}                                                                                                                  \n"

		"void main()                                                                                                        \n"
		"{                                                                                                                  \n"
		"       FragColor =  vec4(0);                                                                                       \n"
		"       int i = int(gl_FragCoord.x);                                                                                \n"
		"       FragColor.r = rand(vec2( (i+TexCoord.x) * sin(no_weight) , (sin(TexCoord.x) * tan(TexCoord.y))/(i+1) ) );   \n"
		"}                                                                                                                  \0"
		;

	const char* WeightUpdateShader_code =
		"#version 330 core                                                                          \n"
		"precision highp float;                                                                     \n"
		"uniform sampler2D Weights;                                                                 \n"
		"uniform sampler2D NeuronsOutput;                                                           \n"
//		"uniform sampler2D ActualOutput;                                                            \n"
		"uniform sampler2D PreviousLayer;                                                           \n"
		"uniform int PreviousLayer_size;                                                            \n"
//		"uniform float OutputLayer_size;                                                            \n"
		"uniform float LearningRate = 1;                                                            \n"
		"out vec4 FragColor;                                                                        \n"
		"in vec2 TexCoord;                                                                          \n"
		"in vec4 gl_FragCoord;                                                                      \n"
		"vec4 weight;                                                                               \n"
//		"int belong_to;                                                                             \n"
		"vec4 output;                                                                               \n"
//		"vec4 A_output;                                                                             \n"
		"float NeuronSelect;                                                                        \n"
		"vec4 inputFrag;                                                                            \n"

		"void main()                                                                                \n"
		"{                                                                                          \n"
		        //get weight
		"       weight =  texture(Weights, TexCoord) ;                                              \n"
		        //get that output neuron value
		"       output = texture(NeuronsOutput, TexCoord);                                          \n"
		        //figure out which input neuron this weight belongs to
		"       NeuronSelect = mod(gl_FragCoord.x, PreviousLayer_size +1);                          \n"
		        //get that input neuron value
		"       inputFrag = texture(PreviousLayer, vec2(NeuronSelect / PreviousLayer_size, 0) );    \n"
		"       if(NeuronSelect == PreviousLayer_size + 0.5)                                        \n"
		"       {    inputFrag.r = 1.0;}                                                             \n"
		        //update weight
		"       weight.r += output.b * inputFrag.r * LearningRate;                                  \n"
		"       FragColor.r = weight.r;                                                             \n"
		"}                                                                                          \0"
		;

	const char* ErrorGen_code =
		"#version 330 core                                         \n"
		"precision highp float;                                    \n"
		"uniform sampler2D NeuronsOutput;                          \n"
		"uniform sampler2D ActualOutput;                           \n"
		"out vec4 FragColor;                                       \n"
		"in vec2 TexCoord;                                         \n"
		"in vec4 gl_FragCoord;                                     \n"

		"float sigmoid_derivative(float val)                       \n"
		"{                                                         \n"
		"       return val * (1.0 - val);                          \n"
		"}                                                         \n"

		"void main()                                               \n"
		"{                                                         \n"
		"       FragColor = texture(NeuronsOutput, TexCoord);      \n"
		"       vec4 A_Output = texture(ActualOutput, TexCoord);   \n"
		"       FragColor.b = A_Output.r - FragColor.r;            \n"
        "       FragColor.b *= sigmoid_derivative(FragColor.r);    \n"
		"}                                                         \0"
		;

	const char* ErrorBackPropogate_code =
		"#version 330 core                                                                                                               \n"
		"precision highp float;                                                                                                          \n"
		"uniform sampler2D NeuronsOutput;                                                                                                \n"
		"uniform sampler2D NextLayerOutput;                                                                                              \n"
		"uniform sampler2D WeightsToNextLayer;                                                                                           \n"
		"uniform int OutputLayer_size;                                                                                                   \n"
		"uniform int NextLayer_size;                                                                                                     \n"
		"out vec4 FragColor;                                                                                                             \n"
		"in vec2 TexCoord;                                                                                                               \n"
		"in vec4 gl_FragCoord;                                                                                                           \n"
		"vec4 nextLayerNeuron;                                                                                                           \n"
		"vec4 weight;                                                                                                                    \n"
		"float ERROR = 0;                                                                                                                \n"
		
		"float getCoord(float index, int size)                                                                                           \n"
		"{                                                                                                                               \n"
		"       return (2*index +1)/ (2*size);                                                                                           \n"
		"}                                                                                                                               \n"
		
		"float sigmoid_derivative(float val)                                                                                             \n"
		"{                                                                                                                               \n"
		"       return val * (1.0 - val);                                                                                                \n"
		"}                                                                                                                               \n"
		
		"void main()                                                                                                                     \n"
		"{                                                                                                                               \n"
		"       int weight_size = (OutputLayer_size +1) * NextLayer_size;                                                                \n"
		"       FragColor = texture(NeuronsOutput, TexCoord);                                                                            \n"
		
		"       for(int i=0; i<OutputLayer_size-1; i++)                                                                                  \n"
		"       {                                                                                                                        \n"
		"           nextLayerNeuron = texture(NextLayerOutput, vec2(getCoord(i,OutputLayer_size),0));                                    \n"
		"           weight = texture(WeightsToNextLayer, vec2(getCoord(i*(OutputLayer_size+1) + int(gl_FragCoord.x),weight_size),0));    \n"

//#ifndef REMOVE_GRADIENT_DESCENT


// ****   below are 3 different ways to backpropogate. 1. is not good but needs to be tested. 2 and 3 are same but in 3 sigmoid_derivation is multiplied latter.
// ****                                                                                       needs to find out which one to use 2/3 if output is going to more than one neuron


// /*1*/"           ERROR += (nextLayerNeuron.b * weight.r) * sigmoid_derivative(nextLayerNeuron.r);                                   \n"
   /*2*/"           ERROR += sigmoid_derivative(FragColor.r) * (nextLayerNeuron.b * weight.r);                                         \n"
// /*3*/"           ERROR += (nextLayerNeuron.b * weight.r);                                                                           \n"

//#endif


/* #ifdef REMOVE_GRADIENT_DESCENT
        "           ERROR += (nextLayerNeuron.b * weight.r) * (nextLayerNeuron.r);                                                       \n"
#endif */

		"       }                                                                                                                        \n"
// /*3*/"       FragColor.b = ERROR * sigmoid_derivative(FragColor.r);                                                                   \n"
        "       FragColor.b = ERROR;                                                                                                     \n"
		"}                                                                                                                               \0"
		;
																																		
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////













//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////  DEFINITIONS  ////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////// public functions & structures /////////////////////////////////////////////////////////////////

struct          NeuralNetwork
{
    unsigned int no_layers = 2;
    unsigned int no_of_input = 0;
    unsigned int no_of_output = 0;
    unsigned int total_weights = 0;
    HermesNetwork::networkType netType;
    HermesNetwork::Layer* inputLayer = nullptr;
    HermesNetwork::Layer* outputLayer = nullptr;
    //unsigned int inputTex = -1;
    GLuint pbo;
};

void            InitNeuralLink()
{
	using namespace HermesNetwork;

    #ifndef HN_SharedGLcontext
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
    #endif


	int success;
	char infoLog[512];

	/* Create And Compile Shaders */
	int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShader_code, NULL);
	
	int WeightInitfragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(WeightInitfragmentShader, 1, &WeightInitShader_code, NULL);
	
	int sig_ActivationfragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(sig_ActivationfragmentShader, 1, &sigmoidActivationShader_code, NULL);

	int WeightUpdatefragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(WeightUpdatefragmentShader, 1, &WeightUpdateShader_code, NULL);

	int ErrorGenfragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(ErrorGenfragmentShader, 1, &ErrorGen_code, NULL);

	int ErrorBPfragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(ErrorBPfragmentShader, 1, &ErrorBackPropogate_code, NULL);

	glCompileShader(vertexShader);
	glCompileShader(WeightInitfragmentShader);
	glCompileShader(sig_ActivationfragmentShader);
	glCompileShader(WeightUpdatefragmentShader);
	glCompileShader(ErrorGenfragmentShader);
	glCompileShader(ErrorBPfragmentShader);

	/* handle compile error	*/
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);

	glGetShaderiv(WeightInitfragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
		glGetShaderInfoLog(WeightInitfragmentShader, 512, NULL, infoLog);

	glGetShaderiv(sig_ActivationfragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
		glGetShaderInfoLog(sig_ActivationfragmentShader, 512, NULL, infoLog);

	glGetShaderiv(WeightUpdatefragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
		glGetShaderInfoLog(WeightUpdatefragmentShader, 512, NULL, infoLog);

	glGetShaderiv(ErrorGenfragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
		glGetShaderInfoLog(ErrorGenfragmentShader, 512, NULL, infoLog);

	glGetShaderiv(ErrorBPfragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
		glGetShaderInfoLog(ErrorBPfragmentShader, 512, NULL, infoLog);

	WeightInit = glCreateProgram();
		glAttachShader(WeightInit, vertexShader);
		glAttachShader(WeightInit, WeightInitfragmentShader);
	glLinkProgram(WeightInit);

	SigmoidActivation = glCreateProgram();
		glAttachShader(SigmoidActivation, vertexShader);
		glAttachShader(SigmoidActivation, sig_ActivationfragmentShader);
	glLinkProgram(SigmoidActivation);

	WeightUpdate = glCreateProgram();
		glAttachShader(WeightUpdate, vertexShader);
		glAttachShader(WeightUpdate, WeightUpdatefragmentShader);
	glLinkProgram(WeightUpdate);

	ErrorGen = glCreateProgram();
		glAttachShader(ErrorGen, vertexShader);
		glAttachShader(ErrorGen, ErrorGenfragmentShader);
	glLinkProgram(ErrorGen);

	ErrorBackPropogate = glCreateProgram();
		glAttachShader(ErrorBackPropogate, vertexShader);
		glAttachShader(ErrorBackPropogate, ErrorBPfragmentShader);
	glLinkProgram(ErrorBackPropogate);

	/* Create Drawing Polygon in VBO */	
	float vertices[] = {
		// positions          // colors           // texture coords
		 1.0f,  1.0f, 0.0f,   0, 0, 0,   1.0f, 1.0f,   // top right
		 1.0f, -1.0f, 0.0f,   0, 0, 0,   1.0f, 0.0f,   // bottom right
		-1.0f, -1.0f, 0.0f,   0, 0, 0,   0.0f, 0.0f,   // bottom left
		-1.0,   1.0f, 0.0f,   0, 0, 0,   0.0f, 1.0f    // top left 
	};
	unsigned int indices[] = {
			0, 1, 3, // first triangle
			1, 2, 3  // second triangle
	};
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// texture coord attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);


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
	glUseProgram(WeightInit);
	WINT_unifm_no_of_weight = glGetUniformLocation(WeightInit, "no_weight");
	glUseProgram(SigmoidActivation);
	SIGACT_unifm_prev_size = glGetUniformLocation(SigmoidActivation, "PreviousLayer_size");
	SIGACT_unifm_weight_size = glGetUniformLocation(SigmoidActivation, "Weight_size");
	SIGACT_unifm_prev_L_TEX = glGetUniformLocation(SigmoidActivation, "PreviousLayer");
	SIGACT_unifm_Layer_weight = glGetUniformLocation(SigmoidActivation, "LayerWeight");	
	glUseProgram(WeightUpdate);
	WGHTUP_unifm_prev_size = glGetUniformLocation(WeightUpdate, "PreviousLayer_size");
	WGHTUP_unifm_LearnRT = glGetUniformLocation(WeightUpdate, "LearningRate");
	WGHTUP_unifm_weight_TEX = glGetUniformLocation(WeightUpdate, "Weights");
	WGHTUP_unifm_neuronOut_TEX = glGetUniformLocation(WeightUpdate, "NeuronsOutput");
	WGHTUP_unifm_prev_L_TEX = glGetUniformLocation(WeightUpdate, "PreviousLayer");
	glUseProgram(ErrorGen);
	ERROR_unifm_neuronOut_TEX = glGetUniformLocation(ErrorGen, "NeuronOutput");
	ERROR_unifm_actualOut_TEX = glGetUniformLocation(ErrorGen, "ActualOutput");
	glUseProgram(ErrorBackPropogate);
	ERROR_BP_unifm_neuronOut_TEX = glGetUniformLocation(ErrorBackPropogate, "NeuronsOutput");
	ERROR_BP_unifm_next_L_TEX = glGetUniformLocation(ErrorBackPropogate, "NextLayerOutput");
	ERROR_BP_unifm_weight_TEX = glGetUniformLocation(ErrorBackPropogate, "WeightsToNextLayer");
	ERROR_BP_unifm_Layer_size = glGetUniformLocation(ErrorBackPropogate, "OutputLayer_size");
	ERROR_BP_unifm_next_L_size = glGetUniformLocation(ErrorBackPropogate, "NextLayer_size");

	glDeleteShader(vertexShader);
	glDeleteShader(WeightInitfragmentShader);
	glDeleteShader(sig_ActivationfragmentShader);
	glDeleteShader(WeightUpdatefragmentShader);
	glDeleteShader(ErrorGenfragmentShader);
	glDeleteShader(ErrorBPfragmentShader);
}

template <typename T = int>
NeuralNetwork*  NetworkBuilder(int InputSize, std::initializer_list<T> HiddenLayers, int OutputSize)
{
    using namespace HermesNetwork;
    NeuralNetwork* nn = createNetwork(InputSize, OutputSize);
    for (int i : HiddenLayers)
        AddLayer(nn, i);
    return nn;
}

void            AddLayer(NeuralNetwork* Network, int size, unsigned int Depth/* = -1*/)
{
	using namespace HermesNetwork;

	Layer* newL = initLayer(size, hiddenL);
	if (Depth > Network->no_layers - 2 || Depth == 0)
		Depth = -1;
	if (Depth == -1)
		connectLayer(Network, newL, Network->outputLayer);
	else
	{
		Layer* next = Network->inputLayer->next;
		for (unsigned int i = 1; i < Depth; i++)
			next = next->next;
		connectLayer(Network, newL, next);
	}
	Network->no_layers++;
}

void            TriggerLayer(NeuralNetwork* Network, int LayerDepth)
{
	using namespace HermesNetwork;

	/* Select Layer at given depth */
	Layer* Lyr = Network->inputLayer;
	for (int i = 0; i < LayerDepth; i++)
		Lyr = Lyr->next;


	/* Trigger selected layer */
	triggerLayer(Lyr);
}

void            TriggerNetwork(NeuralNetwork* Network)
{
	using namespace HermesNetwork;

	Layer* Lyr = Network->inputLayer;
	for (int i = 1; i < Network->no_layers; i++)
	{
		Lyr = Lyr->next;
		triggerLayer(Lyr);
	}
}

void            SendInputs(NeuralNetwork* Network, float Inputs[])
{
	glBindTexture(GL_TEXTURE_2D, Network->inputLayer->NeuronsTex);		
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, Network->inputLayer->no_neuron, 1, 0, GL_RED, GL_FLOAT, Inputs);
	glBindTexture(GL_TEXTURE_2D, 0);
}

float*          GetOutputLayerData(NeuralNetwork* Network)
{
	return HermesNetwork::getLayerNeuronsData(Network, Network->no_layers -1);
}

void            TrainNetwork(NeuralNetwork* Network, float *ActualOutput, float LearningRate /* = 1.0 */)
{
    /*
     *      STEPS FOR BACKPORPAGATION
     *
     *    1.  calculate error for output layer                    --  for each neuron: error = sigmoid_derivative(neuron_output * (ActualOutput-neuron_output))
     *    2.  adjust weights for output layer according to error  --  for each weights coming from neuron m, going to a neuron n: weights += n.error*m.value
     *    3.  calculate error for hidden layer                    --  same as that of output layer error calculation
     *    4.  adjust weights for hidden layer according to error  --  same as that of output layer weight adjustment
     */

	using namespace HermesNetwork;
	// 1.
	calcError(Network->outputLayer, ActualOutput);
	//2.
	trainLayer(Network->outputLayer, &LearningRate);
	//3.
	Layer *Lyr = Network->outputLayer->prev;
	for(int i = Network->no_layers; i > 2; i--)
    {
        backPropogateError(Lyr);
        trainLayer(Lyr, &LearningRate);
        Lyr = Lyr->prev;
    }
//    Layer *Lyr = Network->inputLayer->next;
//    for(int i = Network->no_layers; i > 2; i--)
//    {
//        backPropogateError(Lyr);
//        Lyr = Lyr->next;
//    }
//	Lyr = Network->inputLayer;
//    for(int i=1;i <Network->no_layers;i++)
//    {
//        Lyr = Lyr->next;
//        trainLayer(Lyr, &LearningRate);
//    }

}

void            SaveNetwork(NeuralNetwork* Network, char filename[])
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

    HermesNetwork::Layer *L = Network->inputLayer->next;
    for(int i=1; i < Network->no_layers-1 ; i++,L = L->next)
    {
        file.write((char*)&L->no_neuron, sizeof(int));
        float *weights = HermesNetwork::getWeights_Bias(Network,i);
        file.write((char*)weights, sizeof(float) * L->no_weight);
    }

    file.write((char*)HermesNetwork::getWeights_Bias(Network,Network->no_layers-1), sizeof(float) * Network->outputLayer->no_weight);
    file.close();



}

NeuralNetwork*  LoadNetwork(char filename[])
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

    NeuralNetwork *Network = HermesNetwork::createNetwork(inputSize, outputSize);
    HermesNetwork::Layer *L;

    int hlSize = 0;
    L = Network->inputLayer;
    for(int i = 1; i < layerSize-1; i++)
    {
        file.read((char*)&hlSize,sizeof(int));
        AddLayer(Network,hlSize);
        L = L->next;


        float hlWeights[L->no_weight];
        file.read((char*)hlWeights, sizeof(float) * L->no_weight);

        glBindTexture(GL_TEXTURE_2D, L->WeightTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, L->no_weight, 1, 0, GL_RED, GL_FLOAT, hlWeights);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    L = Network->outputLayer;
    float outputWeights[L->no_weight];
    file.read((char*)outputWeights, sizeof(float) * L->no_weight);
    glBindTexture(GL_TEXTURE_2D, L->WeightTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, L->no_weight, 1, 0, GL_RED, GL_FLOAT, outputWeights);
    glBindTexture(GL_TEXTURE_2D, 0);

    file.close();
    return Network;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////// HermesNetwork's functions /////////////////////////////////////////////////////////////////////
HermesNetwork::Layer*	HermesNetwork::initLayer(int size, layerType typ)
{
	/* Create Layer object */
	Layer* newL = new Layer();
	newL->type = typ;
	newL->no_neuron = size;
	

	/* create frame and render buffer for storing neuron activation values */
	glGenFramebuffers(1, &newL->NeuronsFbo);
	glBindFramebuffer(GL_FRAMEBUFFER, newL->NeuronsFbo);
	glGenTextures(1, &newL->NeuronsTex);
	glBindTexture(GL_TEXTURE_2D, newL->NeuronsTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, newL->no_neuron, 1, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, newL->NeuronsTex, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	// OLD: glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, newL->NeuronsTex, 0);
	

	
	/* create frame buffer for weight */
	if (typ != inputL)
		glGenFramebuffers(1, &newL->WeightFbo);
	/*
	* NOTE: genreating WeightTex and linking WeightFbo to WeightTex is done latter when two layers are connected
	*		Because right now we dont know the no. of neurons of previous layers.
	*		So, we cannot create a WeightTex of (previous->no_nourons + 1) * newL->no_nueorns here.
	*/

	glBindTexture(GL_TEXTURE_2D, 0);
	return newL;
}

NeuralNetwork*			HermesNetwork::createNetwork(int InputSize, int OutputSize)
{

	/* build input Layer structure */
	Layer* inp = initLayer(InputSize, inputL);

	/* build output Layer structure */
	Layer* op = initLayer(OutputSize, outputL);


	/* build NeuralNetowkr structure */
	NeuralNetwork* nn = new NeuralNetwork();
	nn->no_of_input = InputSize;
	nn->no_of_output = OutputSize;
	nn->inputLayer = inp;
	nn->outputLayer = op;
	connectLayer(nn, inp, op);



	/*glGenTextures(1, &nn->inputTex);
	glBindTexture(GL_TEXTURE_2D, nn->inputTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F,InputSize , 1, 0, GL_RGB, GL_FLOAT, 0);*/

	/*
	* cannot do this as it will initialize weights in netork with only input and output layer initialy
	* that means it will have to use initWeight_adderLayer() from addLayer()
	* cant use initWeight_adderLayer() till its problem is solved, go to initWeight_adderLayer() to see problem

	initWeights(nn);
	*/

	/* Create pbo to retrive neuron data and weight */
	glGenBuffers(1, &nn->pbo);


	glBindTexture(GL_TEXTURE_2D, 0);
	return nn;
}

void					HermesNetwork::connectLayer(NeuralNetwork* Network, Layer* prev, Layer* next)
{
	//build chain
	/*TODO
	* check and delete next->WeightTex and generate new one as per previous adder layer no. of neurons
	* vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	*/	

	if (next->WeightTex > 0)  //delete old texture RBO if next layer already connected to an old previous layer
	{
		glDeleteTextures(1, &next->WeightTex);
		Network->total_weights -= next->no_weight;
	}
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	if (prev->next != nullptr)
		connectLayer(Network, next, prev->next);

	prev->next = next;
	if (next->prev != nullptr)
	{
		prev->prev = next->prev;
		prev->prev->next = prev;
	}
	next->prev = prev;

	/* init next's weight texture */
	next->no_weight = (prev->no_neuron + 1) * next->no_neuron;
	glBindFramebuffer(GL_FRAMEBUFFER, next->WeightFbo);
	glGenTextures(1, &next->WeightTex);
	glBindTexture(GL_TEXTURE_2D, next->WeightTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, next->no_weight, 1, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);	
	//OLD: glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, next->WeightTex, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, next->WeightTex, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	

	Network->total_weights += next->no_weight;
	

	/* init next's weights to random value(0.5<->1.5) */	
	glViewport(0, 0, next->no_weight, 1);
	glBindFramebuffer(GL_FRAMEBUFFER, next->WeightFbo);	
	glUseProgram(HermesNetwork::WeightInit);
	glUniform1i(WINT_unifm_no_of_weight, next->no_weight);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);


	/* if previous layer is input layer skip remaing part as we dont need weights in input layer */
	if (prev->type == inputL)
		return;

	/* init prev's weight texture */
	prev->no_weight = (prev->prev->no_neuron + 1) * prev->no_neuron;
	glBindFramebuffer(GL_FRAMEBUFFER, prev->WeightFbo);
	glGenTextures(1, &prev->WeightTex);
	glBindTexture(GL_TEXTURE_2D, prev->WeightTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, prev->no_weight, 1, 0, GL_RGB, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);		
	//OLD:glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, prev->WeightRbo, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, prev->WeightTex, 0);
	glBindTexture(GL_TEXTURE_2D, 0);


	/* init prev's weights to random value(0.5<->1.5) */
	glViewport(0, 0, prev->no_weight, 1);
	glBindFramebuffer(GL_FRAMEBUFFER, prev->WeightFbo);
	glUseProgram(HermesNetwork::WeightInit);
	glUniform1i(WINT_unifm_no_of_weight, prev->no_weight);	
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	
	Network->total_weights += prev->no_weight;

	/* Unbind fbo and Texture before returning */
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

float*					HermesNetwork::getWeights_Bias(NeuralNetwork* Network, int LayerDepth)
{
	/* Select Layer at given depth */
	Layer* Lyr = Network->inputLayer->next;
	for (int i = 1; i < LayerDepth; i++)
		Lyr = Lyr->next;


	/* Read Buffer texture */
	glBindBuffer(GL_PIXEL_PACK_BUFFER, Network->pbo);
	glBufferData(GL_PIXEL_PACK_BUFFER, Lyr->no_weight * 4 * sizeof(float), 0, GL_STREAM_READ);
	glBindFramebuffer(GL_FRAMEBUFFER, Lyr->WeightFbo);
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glReadPixels(0, 0, Lyr->no_weight, 1, GL_RED, GL_FLOAT, 0);
	//glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	GLfloat* src = (GLfloat*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
	glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
	return src;
}

void					HermesNetwork::triggerLayer(Layer* Lyr)
{
	/* Run Activation Shader */
	//SIGACT_unifm_prev_size = glGetUniformLocation(SigmoidActivation, "PreviousLayer_size");
	//SIGACT_unifm_weight_size = glGetUniformLocation(SigmoidActivation, "Weight_size");
	//SIGACT_unifm_prev_L_TEX = glGetUniformLocation(SigmoidActivation, "PreviousLayer");
	//SIGACT_unifm_Layer_weight = glGetUniformLocation(SigmoidActivation, "LayerWeight");
	glViewport(0, 0, Lyr->no_neuron, 1);
	glBindFramebuffer(GL_FRAMEBUFFER, Lyr->NeuronsFbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Lyr->prev->NeuronsTex);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, Lyr->WeightTex);

	glUseProgram(HermesNetwork::SigmoidActivation);
	glUniform1i(SIGACT_unifm_prev_size, Lyr->prev->no_neuron);
	glUniform1i(SIGACT_unifm_weight_size, Lyr->no_weight);
	glUniform1i(SIGACT_unifm_prev_L_TEX, 0);
	glUniform1i(SIGACT_unifm_Layer_weight, 1);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);


	/* unbind textures and fbo */
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

float*					HermesNetwork::getLayerNeuronsData(NeuralNetwork* Network, int LayerDepth)
{	
	/* Select Layer at given depth */
	Layer* Lyr = Network->inputLayer;
	for (int i = 0; i < LayerDepth; i++)
		Lyr = Lyr->next;

	/* Read Buffer texture */
	glBindBuffer(GL_PIXEL_PACK_BUFFER, Network->pbo);
	glBufferData(GL_PIXEL_PACK_BUFFER, Lyr->no_neuron * 4 * sizeof(float), 0, GL_STREAM_READ);
	glBindFramebuffer(GL_FRAMEBUFFER, Lyr->NeuronsFbo);
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glReadPixels(0, 0, Lyr->no_neuron, 1, GL_RED, GL_FLOAT, 0);
	//glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	GLfloat* src = (GLfloat*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
	glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
	return src;

	///* Store in return array */
	//float* ret = new float[Lyr->no_neuron];
	//for (int i = 0; i < Lyr->no_neuron; i++)
	//	ret[i] = src[i * 3];

	//glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	//return ret;
}

void					HermesNetwork::trainLayer(Layer* Lyr, float* LearningRate)
{
	glViewport(0, 0, Lyr->no_weight, 1);
	
	//bind fbo and all texture data
	glBindFramebuffer(GL_FRAMEBUFFER, Lyr->WeightFbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Lyr->WeightTex);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, Lyr->NeuronsTex);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, Lyr->prev->NeuronsTex);


	//set uniforms and run shader
	glUseProgram(HermesNetwork::WeightUpdate);
	glUniform1i(HermesNetwork::WGHTUP_unifm_prev_size, Lyr->prev->no_neuron);
	glUniform1f(HermesNetwork::WGHTUP_unifm_LearnRT, *LearningRate);
	glUniform1i(HermesNetwork::WGHTUP_unifm_weight_TEX, 0);
	glUniform1i(HermesNetwork::WGHTUP_unifm_neuronOut_TEX, 1);
	glUniform1i(HermesNetwork::WGHTUP_unifm_prev_L_TEX, 2);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);


	//unbind fbo and all texture data
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void					HermesNetwork::calcError(Layer* Lyr, float* ActualOutput)
{
	/* convert output array to texture */
	glBindTexture(GL_TEXTURE_2D, TempTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, Lyr->no_neuron, 1, 0, GL_RED, GL_FLOAT, ActualOutput);

	/* bind all input texture for shader */
	glViewport(0, 0, Lyr->no_neuron, 1);
	glBindFramebuffer(GL_FRAMEBUFFER, Lyr->NeuronsFbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Lyr->NeuronsTex);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, TempTex);
	glUseProgram(ErrorGen);
	glUniform1i(ERROR_unifm_neuronOut_TEX, 0);
	glUniform1i(ERROR_unifm_actualOut_TEX, 1);

	/* run shader */
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	/* unbind textures and fbo */
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void					HermesNetwork::backPropogateError(Layer* Lyr)
{
	/* bind all input texture for shader */
	glViewport(0, 0, Lyr->no_neuron, 1);
	glBindFramebuffer(GL_FRAMEBUFFER, Lyr->NeuronsFbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Lyr->NeuronsTex);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, Lyr->next->NeuronsTex);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, Lyr->next->WeightTex);
	glUseProgram(ErrorBackPropogate);
	glUniform1i(ERROR_BP_unifm_neuronOut_TEX, 0);
	glUniform1i(ERROR_BP_unifm_next_L_TEX, 1);
	glUniform1i(ERROR_BP_unifm_weight_TEX, 2);
	glUniform1i(ERROR_BP_unifm_Layer_size, Lyr->no_neuron);
	glUniform1i(ERROR_BP_unifm_next_L_size, Lyr->next->no_neuron);

	/* run shader */
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	/* unbind textures and fbo */
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif // !__HERMES_NETWORK__