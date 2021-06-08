#ifndef __HERMES_NETWORK__
#define __HERMES_NETWORK__



//////////////////////////////////////////////////// Delaration /////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//Handle to entire network.
//[Always create a pointer object of it]
struct NeuralNetwork;

//Setup gl context, compile shaders, create drawing polygon, crate ssbo
void initNeuralLink();

//Build neural network with given input size, hidden layers sizes as array and output size
template <size_t n>	NeuralNetwork*
NetworkBuilder(int inputSize, int(&HiddenLayers)[n], int OutputSize);

//Creates network with only input and output. Returns a pointer of type- NeuralNetwork
NeuralNetwork* CreateNetwork(int inputSize, int OutputSize);

//Adds a new layer at given depth. No depth or negative depth will add layer at last before output layer
void addLayer(NeuralNetwork* Network, int size, unsigned int Depth);

//Returns array of data of every neuron from layer at specified depth
float* getLayerNeuronsData(NeuralNetwork* Network, int LayerDepth);

//Activates every neurons of a layer at specified depth
void triggerLayer(NeuralNetwork* Network, int LayerDepth);

//Set every neurons of input layer with specified values in array
void SendInputs(NeuralNetwork* Network, float Inputs[]);


//HermesNetwork library builds neural network inside GPU and triggers netwrok layers as per command.
//The Library uses OpenGL to access GPUs. It can be run on any dedicated or intergrated GPU regardless of GPU vendor's drivers, OS or hardware architecture.
//It provides features to add layers at any depth, send input data to input layer, retrive data or weights from any layer.
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

	GLFWwindow* offscreen_context;
	unsigned int VBO;
	unsigned int VAO;
	unsigned int EBO;
	int WeightInit;
	int WeightUpdate;
	int SigmoidActivation;
	GLint WINT_unifm_no_of_weight;
	GLint SIGACT_unifm_prev_size;


	///////////////////////////////// Functions ////////////////////////////////////////////////

	//This function will be called whenever a new layer is created
	Layer* initLayer(int size, layerType typ);

	//Connect two layer with weight
	void connectLayer(NeuralNetwork* Network, Layer* prev, Layer* next);

	//Initialize weights of all layer in network with random value between 0.5 and 1.5 in WeightBuffer of each layer. 
	void initWeights(NeuralNetwork* Network);

	//Returns an array of all the weights (along with bias after list of weight of a neuron) of layer at the specified depth
	float* getWeights_Bias(NeuralNetwork* Network, int LayerDepth);


	////////////////////////////////////////////// Shader Codes ////////////////////////////////////////////////////
	const char* vertexShader_code =
		"#version 330 core												\n"
		"layout(location = 0) in vec3 aPos;								\n"
		"layout(location = 1) in vec3 aColor;							\n"
		"layout(location = 2) in vec2 aTexCoord;						\n"
		"out vec2 TexCoord;												\n"
		"void main()													\n"
		"{																\n"
		"	gl_Position = vec4(aPos.x, -aPos.y, aPos.z  , 1.0);			\n"
		"   TexCoord = vec2(aTexCoord.x, aTexCoord.y);					\n"
		"}																\0"
		;

	const char* sigmoidActivationShader_code =
		"#version 330 core													\n"
		"precision highp float;												\n"
		"uniform int PreviousLayer_size; \n"
		"uniform sampler2D PreviousLayer;									\n"
		"uniform sampler2D LayerWeight;										\n"		
		"in vec2 TexCoord;													\n"
		"in vec4 gl_FragCoord;												\n"
		"out vec4 FragColor;												\n"		

		"float sigmoid(float x)												\n"
		"{																	\n"
		"		return 1.0/(1.0 + exp(-x));									\n"
		"}																	\n"

		"void main()														\n"
		"{																	\n"
		"		FragColor = vec4(0);										\n"
		//"		int start = int(gl_FragCoord.x) * (no_prevL_neurons + 1);	\n"
		//"		int end = start + no_prevL_neurons;							\n"
		//"		float val = 0;												\n"
		//"		vec4 pxl;													\n"
		//"		for(int i=0; i< no_prevL_neurons; i++)						\n"
		//"		{															\n"
		//"			pxl = texture(PreviousLayer, vec2(i,0));				\n"
		//"			val += pxl.r;											\n"
		//"			pxl = texture(LayerWeight, vec2(start + i,0));			\n"
		//"			val *= pxl.r;											\n"
		//"		}															\n"
		//"		pxl = texture(LayerWeight, vec2(end, 0));					\n"
		//"		val += pxl.r;												\n"
		//"		FragColor.r = sigmoid(val);									\n"
		"		FragColor.r = -PreviousLayer_size;\n"
		"}\0"
		;

	const char* WeightInitShader_code =
		"#version 330 core																									\n"
		"precision highp float;																								\n"
		"out vec4 FragColor;																								\n"
		"in vec2 TexCoord;																									\n"
		"uniform int no_weight;																								\n"
//		"uniform sampler2D Texture;																							\n"
		"in vec4 FragCoord;																									\n"

		"float rand(vec2 co)																								\n"
		"{																													\n"
		"    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);												\n"
		"}																													\n"

		"void main()																										\n"
		"{\n"		
		"		FragColor =  vec4(0);																						\n"
		"		int i = int(gl_FragCoord.x);																				\n"		
		"		FragColor.r = rand(vec2( (i+TexCoord.x) * sin(no_weight) , (sin(TexCoord.x) * tan(TexCoord.y))/(i+1) ) );	\n"																							
		"}																													\0"				
		;																																
																																		

	const char* WeightUpdateShader_code =
		"#version 430 core									\n"
		"precision highp float;								\n"
		"in vec2 TexCoord;									\n"
		"in vec4 gl_FragCoord;								\n"
		"uniform sampler2D TexBuffer;						\n"
		"int i = 0;											\n"
		"void main()										\n"
		"{													\n"
		"		i = int(gl_FragCoord.x);					\n"
		"		x[i] = texture(TexBuffer, vec2(i,0));		\n"
		"}													\n"
		;

}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////














/////////////////////////////////////////////////////////////////////////  Defination  ///////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////// public functions & structures //////////////////////////////////////////////
//Handle to entire network.
//[Always create a pointer object of it]
struct NeuralNetwork
{
	unsigned int no_layers = 2;
	unsigned int no_of_input = 0;
	unsigned int no_of_output = 0;
	unsigned int total_weights = 0;
	HermesNetwork::networkType netType;
	HermesNetwork::Layer* inputLayer = nullptr;
	HermesNetwork::Layer* outputLayer = nullptr;
	unsigned int inputTex = -1;
	GLuint pbo;
};


//Setup gl context, compile shaders, create drawing polygon, crate ssbo
void initNeuralLink()
{
	using namespace HermesNetwork;

	//Create window context
					//use this when glfw is removed to create window and context
					/*HWND dummyHWND = ::CreateWindowA("STATIC", "dummy", WS_VISIBLE, 0, 0, 100, 100, NULL, NULL, NULL, NULL);
					::SetWindowTextA(dummyHWND, "Dummy Window!");*/
	glfwInit();
	//glfwWindowHint(GLFW_NO_API, GLFW_TRUE);
	//glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	offscreen_context = glfwCreateWindow(50, 50, "", NULL, NULL);
	glfwMakeContextCurrent(offscreen_context);
	glViewport(0, 0, 50, 50);
	glfwSwapBuffers(offscreen_context);
	glfwPollEvents();

	glewInit();

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
	glShaderSource(WeightUpdatefragmentShader, 1, &WeightInitShader_code, NULL);

	glCompileShader(vertexShader);
	glCompileShader(WeightInitfragmentShader);
	glCompileShader(sig_ActivationfragmentShader);
	glCompileShader(WeightUpdatefragmentShader);


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



	WeightInit = glCreateProgram();
		glAttachShader(WeightInit, vertexShader);
		glAttachShader(WeightInit, WeightInitfragmentShader);
	glLinkProgram(WeightInit);

	WeightUpdate = glCreateProgram();
		glAttachShader(WeightInit, vertexShader);
		glAttachShader(WeightInit, WeightInitfragmentShader);
	glLinkProgram(WeightUpdate);

	SigmoidActivation = glCreateProgram();
		glAttachShader(SigmoidActivation, vertexShader);
		glAttachShader(SigmoidActivation, sig_ActivationfragmentShader);
	glLinkProgram(SigmoidActivation);



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



	/* Get all uniform locations */
	glUseProgram(WeightInit);
	WINT_unifm_no_of_weight = glGetUniformLocation(WeightInit, "no_weight");
	glUseProgram(SigmoidActivation);
	SIGACT_unifm_prev_size = glGetUniformLocation(SigmoidActivation, "PreviousLayer_size");

	glDeleteShader(vertexShader);
	glDeleteShader(WeightInitfragmentShader);
	glDeleteShader(sig_ActivationfragmentShader);
}


//Creates network with only input and output. Returns a pointer of type- NeuralNetwork
NeuralNetwork* CreateNetwork(int inputSize, int OutputSize)
{
	using namespace HermesNetwork;

	/* build input Layer structure */
	Layer* inp = initLayer(inputSize, inputL);

	/* build output Layer structure */
	Layer* op = initLayer(OutputSize, outputL);


	/* build NeuralNetowkr structure */
	NeuralNetwork* nn = new NeuralNetwork();
	nn->no_of_input = inputSize;
	nn->no_of_output = OutputSize;
	nn->inputLayer = inp;
	nn->outputLayer = op;
	connectLayer(nn, inp, op);

	

	glGenTextures(1, &nn->inputTex);
	glBindTexture(GL_TEXTURE_2D, nn->inputTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F,inputSize , 1, 0, GL_RGB, GL_FLOAT, 0);

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


//Adds a new layer at given depth. No depth or negative depth will add layer at last before output layer
void addLayer(NeuralNetwork* Network, int size, unsigned int Depth = -1)
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


//Returns array of data of every neuron from layer at specified depth
float* getLayerNeuronsData(NeuralNetwork* Network, int LayerDepth)
{
	using namespace HermesNetwork;

	/* Select Layer at given depth */
	Layer* Lyr = Network->inputLayer;
	for (int i = 0; i < LayerDepth; i++)
		Lyr = Lyr->next;

	/* Read Buffer texture */
	glBindBuffer(GL_PIXEL_PACK_BUFFER, Network->pbo);
	glBufferData(GL_PIXEL_PACK_BUFFER, Lyr->no_neuron * 4 * sizeof(float), 0, GL_STREAM_READ);
	glBindFramebuffer(GL_FRAMEBUFFER, Lyr->NeuronsFbo);
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glReadPixels(0, 0, Lyr->no_neuron, 1, GL_RGB, GL_FLOAT, 0);
	//glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	GLfloat* src = (GLfloat*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
	glUnmapBuffer(GL_PIXEL_PACK_BUFFER);


	/* Store in return array */
	float* ret = new float[Lyr->no_neuron];
	for (int i = 0; i < Lyr->no_neuron; i++)
		ret[i] = src[i * 3];

	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	return ret;
}


//Activates every neurons of a layer at specified depth
void triggerLayer(NeuralNetwork* Network, int LayerDepth)
{
	using namespace HermesNetwork;

	/* Select Layer at given depth */
	Layer* Lyr = Network->inputLayer;
	for (int i = 0; i < LayerDepth; i++)
		Lyr = Lyr->next;


	/* Run Activation Shader */
	glViewport(0, 0, Lyr->no_neuron, 1);
	glBindFramebuffer(GL_FRAMEBUFFER, Lyr->NeuronsFbo);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Lyr->prev->NeuronsTex);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, Lyr->WeightTex);
	glUniform1i(SIGACT_unifm_prev_size, Lyr->no_weight);
	glUseProgram(HermesNetwork::SigmoidActivation);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, 0);
}


//Set every neurons of input layer with specified values in array
void SendInputs(NeuralNetwork* Network, float Inputs[])
{
	glBindTexture(GL_TEXTURE_2D, Network->inputLayer->NeuronsTex);		
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, Network->inputLayer->no_neuron, 1, 0, GL_RED, GL_FLOAT, Inputs);
	glBindTexture(GL_TEXTURE_2D, 0);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////// HermesNetwork's functions ////////////////////////////////////////////////////////////
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
	glBindTexture(GL_TEXTURE_2D, 0);
	// OLD: glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, newL->NeuronsTex, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, newL->NeuronsTex, 0);

	
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
	glUniform1i(WINT_unifm_no_of_weight, next->no_weight);
	glUseProgram(HermesNetwork::WeightInit);
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


	/* init next's weights to random value(0.5<->1.5) */
	glViewport(0, 0, prev->no_weight, 1);
	glBindFramebuffer(GL_FRAMEBUFFER, prev->WeightFbo);
	glUniform1i(WINT_unifm_no_of_weight, prev->no_weight);
	glUseProgram(HermesNetwork::WeightInit);
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
	glReadPixels(0, 0, Lyr->no_weight, 1, GL_RGB, GL_FLOAT, 0);
	//glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	GLfloat* src = (GLfloat*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
	glUnmapBuffer(GL_PIXEL_PACK_BUFFER);


	/* Store in return array */
	float* ret = new float[Lyr->no_weight];
	for (int i = 0; i < Lyr->no_weight; i++)
		ret[i] = src[i * 3];


	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	return ret;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // !__HERMES_NETWORK__