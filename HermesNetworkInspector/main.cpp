#include<CleanImGuiWin.h>
#include <iostream>
#include <vector>
#include "HermesNetwork.h"

void Theme1();
void DrawPingPong();
void DrawLogicGateTrainer();

void CreateNeuralNetwork();
void DeleteNeuralNetwork();

namespace PingPong
{
	bool Enable = false;
	float plr1Pos = 90;
	float plr2Pos = 90;
	float speed = 1.8f;
	float ballspeed = 6.0;
	float ballangle = ((int)glfwGetTime()) % 180;
	ImVec2 ballpos = { 400,300 };
	bool pause = false;
	int scoreboard[2] = { 0,0 };


	NeuralNetwork N1, N2;
	HermesNetwork::Layer L;
	float actualOUT[] = { 0,0 }, input[3];
	//float* out1, * out2;
	float LR = 1.8, ACK = 0.2;
	bool nn1Learn = true, nn2Learn= true;
}

namespace LogicGateTrainer {
	bool enable = false, end = false;
	bool andTrain = false;
	bool orTrain = false;
	bool xorTrain = false;
	int hiddenLayerSize = 0;
	std::vector<int> hiddenLayers;
	bool networkBuilt = false;

	bool TT[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
	bool AND[4] = {0,0,0,1};
	bool OR[4] = {0,1,1,1};
	bool XOR[4] = {0,1,1,0};
	int inpIndex = 0;

	float Result[4][3];
}

NeuralNetwork NN = NULL;
int Input_OutputLayer[2] = { 1,1 };
int hiddenLayerSize = 0;
std::vector<int> hiddenLayers;
uint8_t activationType = ActivationType::Sigmoid;

float* INPUTdata = NULL, * OUTPUTdata = NULL, LR = 1;

bool CTRLPNL = false;

ImVec2 NNStructSize = { 300,320 }, NNCTRLsize = { 300, 300 };
ImVec2 WinPOS, BTNsize = { 280,22 };

float TexLYRparts = 0;

int main() {
	GLFWwindow* window = ImGui::initGLFW(1000, 800);
	ImGui::initImGui(window, "HermesNetwork Inspector");
	ImGui::GetIO().IniFilename = NULL;

	GLFWmonitor* primary = glfwGetPrimaryMonitor();
	const GLFWvidmode* mode = glfwGetVideoMode(primary);
    int swapInterval = mode->refreshRate/60;
    if(swapInterval < 1)
        swapInterval = 1;
	glfwSwapInterval(swapInterval);

	char FILE[50] = "";	
	
	InitNeuralLink(true);
	
	Theme1();
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 15.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.0f);			

	while (!glfwWindowShouldClose(window)) {
		ImGui::StartCleanWindow(window);				

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {4,3});
		ImGui::Begin("###THEME", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoTitleBar);			
			ImGui::SetWindowPos({ 74, -4 });
			/*ImGui::SetWindowPos({ 54, -4 });*/
			if (ImGui::Button("1"))	Theme1();
			ImGui::SameLine();
			if (ImGui::Button("2")) ImGui::StyleColorsDark();
			ImGui::SameLine();
			if (ImGui::Button("Ping Pong Demo")) {
				if(!PingPong::Enable) {
					PingPong::Enable = true;
					PingPong::N1 = NetworkBuilder(3, {50}, 2);
					PingPong::N2 = NetworkBuilder(3, {}, 2);
					Terrify(PingPong::N1);
					Terrify(PingPong::N2);
					PingPong::scoreboard[0] = PingPong::scoreboard[1] = 0;
					PingPong::speed = 1.8f;
					PingPong::ballspeed = 6.0;
				}
			}
            ImGui::SameLine();
            if(ImGui::Button("Logic Gates Trainer")) {
				LogicGateTrainer::enable = true;
            }
		ImGui::End();
		ImGui::PopStyleVar();

		//neural network structure window
		ImGui::SetNextWindowPos({ 2, ImGui::GetCursorPosY() });
		ImGui::Begin("Neural Network Structure", NULL ,ImGuiWindowFlags_NoResize);
		{
			ImGui::SetWindowSize(NNStructSize);
			if (!CTRLPNL) {
				ImGui::Text("Input & Output Layers");
				ImGui::DragInt2("##inp_out", Input_OutputLayer);
				ImGui::Text("Hidden Layers");
				if(ImGui::InputInt("size", &hiddenLayerSize, 1, 1)) {
					if(hiddenLayerSize < 0)
						hiddenLayerSize = 0;
					if(hiddenLayerSize < hiddenLayers.size())
						hiddenLayers.pop_back();
					if(hiddenLayerSize > hiddenLayers.size())
						hiddenLayers.push_back(1);
				}
				if(ImGui::BeginChild("hl child", { 0,100 }, true)) {
					for (int i = 0; i < hiddenLayers.size(); i++) {
						ImGui::PushID(i);							
						ImGui::DragInt("", &hiddenLayers[i],1);
						ImGui::SameLine();
						ImGui::Text("%d",i+1);
						ImGui::PopID();
					}
					ImGui::EndChild();
				}
				ImGui::Separator();
				if (ImGui::Button("Create Neural Network", BTNsize))
				{					
					CreateNeuralNetwork();
				}
				
				if (ImGui::CollapsingHeader("Load Neural Network"))
				{
					ImGui::InputText("", FILE, 50);
					ImGui::SameLine();
					if (ImGui::Button("Load") && FILE[0] != '\0')
					{
						NN = LoadNetwork(FILE);
						std::fill_n(FILE, 50, '\0');

						Input_OutputLayer[0] = NN->inputLayer->no_neuron;
						Input_OutputLayer[1] = NN->outputLayer->no_neuron;
						INPUTdata = new float[Input_OutputLayer[0]]();
						OUTPUTdata = new float[Input_OutputLayer[1]]();			
						hiddenLayerSize = NN->no_layers - 2;
						CTRLPNL = true;
						TexLYRparts = WinPOS.y / ((hiddenLayerSize + 2) * 2 - 1) - 20;

						TriggerNetwork(NN);
					}
				}
			}
			
			else
			{
				ImGui::Text("Input: %d", Input_OutputLayer[0]);
				ImGui::Text("Hidden: %d", hiddenLayers.size());
				if(!hiddenLayers.empty()) {
				ImGui::BeginChild("HddnL", { 0 ,40 }, true, ImGuiWindowFlags_HorizontalScrollbar);
					for(int sz: hiddenLayers) {
						ImGui::SameLine();
						ImGui::Text("%d  ", sz);
					}
					ImGui::EndChild();
				}
				ImGui::Value("Output", Input_OutputLayer[1]);
				const char* activationNames[] = {"Sigmoid", "TanH", "ReLu"};
				ImGui::Text("Activation: %s", activationNames[activationType]);
				ImGui::Separator();
				if (ImGui::Button("Delete Neural Network", BTNsize))
				{
					DeleteNeuralNetwork();
				}
				if (ImGui::CollapsingHeader("Save Neural Network"))
				{										
					ImGui::InputText("", FILE, 50);
					ImGui::SameLine();
					if (ImGui::Button("Save") && FILE[0] != '\0')
					{
						SaveNetwork(NN, FILE);
						std::fill_n(FILE, 50, '\0');
					}
				}				
			}
		}
		WinPOS = ImGui::GetWindowSize();
		ImGui::End();
		
		WinPOS.x = ImGui::GetWindowSize().y;
		ImGui::SetNextWindowPos({ 2,WinPOS.y + 40 });

		//neural network control panel
		ImGui::Begin("Neural Network Control Panel",NULL, ImGuiWindowFlags_NoResize);
		{
			NNCTRLsize.y = WinPOS.x - ImGui::GetCursorPosY() - WinPOS.y - 30;
			ImGui::SetWindowSize(NNCTRLsize);
			if (CTRLPNL)
			{
				if (ImGui::Button("Trigger Network", BTNsize))
				{
					//call HermisNetwork Trigger
					TriggerNetwork(NN);
				}
				if (ImGui::Button("Train Network", BTNsize))
				{
					// call hermisNetwork Train
					TrainNetwork(NN, OUTPUTdata, LR);
				}
				ImGui::PushItemWidth(180);
				ImGui::DragFloat("Learning Rate", &LR, 0.02f);
				const uint32_t step = 1;
                if(ImGui::InputScalar("Batch Size", ImGuiDataType_U32, &NN->batchSize, &step));
                if(ImGui::Button("Sigmoid")) {
					activationType = ActivationType::Sigmoid;
					SetActivation(NN, ActivationType::Sigmoid);
				}
                ImGui::SameLine();
                if(ImGui::Button("TanH")) {
					activationType = ActivationType::TanH;
					SetActivation(NN, ActivationType::TanH);
				}
                ImGui::SameLine();
                if(ImGui::Button("ReLu")) {
					activationType = ActivationType::ReLu;
					SetActivation(NN, ActivationType::ReLu, ActivationType::Sigmoid);
				}
				ImGui::NewLine();
				ImGui::Separator();
				

				if (ImGui::CollapsingHeader("Network Input")) 
				{
					ImGui::BeginChild("INP child", { NNCTRLsize.x - 50,170 }, true);
					for (int i = 0; i < Input_OutputLayer[0]; i++)
					{
						ImGui::PushID(i);						
						ImGui::DragFloat("", &INPUTdata[i],0.1f);
						ImGui::SameLine();
						ImGui::Text("%d", i);
						ImGui::PopID();
					}
					ImGui::EndChild();
					if (ImGui::Button("Send Inputs", BTNsize))
					{
						SendInputs(NN, INPUTdata);
					}
				}
				
				
				
				if (ImGui::CollapsingHeader("Network Actual Output"))
				{
					ImGui::BeginChild("OUT child", { NNCTRLsize.x - 50,150 }, true);
					for (int i = 0; i < Input_OutputLayer[1]; i++)
					{
						ImGui::PushID(i);
						ImGui::Text("%d", i); ImGui::SameLine();
						ImGui::DragFloat("", &OUTPUTdata[i], 0.01f,0,1);
						ImGui::PopID();
					}
					ImGui::EndChild();					
				}
				
			}
			
		}
		ImGui::End();
		
		
		
		WinPOS.y = ImGui::GetCursorPosY();
		WinPOS.x = 304;
		ImGui::SetNextWindowPos(WinPOS);
		WinPOS = ImGui::GetWindowSize();
		WinPOS.x -= 306;
		WinPOS.y -= 50;
		ImGui::SetNextWindowSize(WinPOS);

		//neural nework live image
		ImGui::Begin("Neural Network Live Image", NULL, ImGuiWindowFlags_NoResize);
		{									
			if (CTRLPNL)
			{								
				WinPOS.x -= 40;
				
				ImGui::Text("I"); ImGui::SameLine();
				ImGui::Image((ImTextureID)NN->inputLayer->NeuronsTex, {WinPOS.x,35});


				HermesNetwork::Layer L = NN->inputLayer;
				
				for (int i = 0; i < hiddenLayers.size(); i++) {
					L = L->next;
					ImGui::Text("w"); ImGui::SameLine();
					ImGui::Image((ImTextureID)L->WeightsTex, { WinPOS.x,15 });
					if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0))
					{						
						ImGui::PushID(-(i + 1));
						ImGui::OpenPopup("");
						ImGui::PopID();
					}
					
					ImGui::Text("%d", i); ImGui::SameLine();
					ImGui::Image((ImTextureID)L->NeuronsTex, { WinPOS.x,35 });
					if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0))
					{
						ImGui::PushID(i + 1);
						ImGui::OpenPopup("");
						ImGui::PopID();
					}
				}

				

				ImGui::Text("w"); ImGui::SameLine();
				ImGui::Image((ImTextureID)NN->outputLayer->WeightsTex, { WinPOS.x,15 });
				if (ImGui::IsItemHovered()  && ImGui::IsMouseClicked(0))
					ImGui::OpenPopup("1p");


				ImGui::Text("O"); ImGui::SameLine();
				ImGui::Image((ImTextureID)NN->outputLayer->NeuronsTex, { WinPOS.x,35 });							
				if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0))
					ImGui::OpenPopup("outL_DT");
	
				L = NN->inputLayer;
				for (int i = 0; i < hiddenLayers.size(); i++) {
					L = L->next;
					ImGui::PushID(-(i+1));
					if (ImGui::BeginPopup(""))
					{
						ImGui::Text("Weights & Bias");
						
						ImGui::Separator();
						if (ImGui::Selectable("Re-Initialize Weights"))
						{
							glBindImageTexture(0, L->WeightsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
							glUseProgram(HermesNetwork::WeightInit);

							glUniform1i(HermesNetwork::WINT_unifm_seed, rand());
							glDispatchCompute(L->no_weight, 1, 1);
							glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
						}
						HermesNetwork::fetchLayerWeights_Bias(L);						
						/*if (NEURONS_DATA == NULL)
							NEURONS_DATA = HermesNetwork::getWeights_Bias(NN, i + 1);*/
						ImGui::BeginChild("HL Weights Data", { 300,40 }, true, ImGuiWindowFlags_HorizontalScrollbar);
						for (int j = 0; j < L->no_weight; j++)
						{
							ImGui::TextColored({ 1,0,0,1 }, "%f ", L->weights[j]);
							ImGui::SameLine();
						}
						ImGui::EndChild();
						ImGui::EndPopup();
						//ImGui::PopID();
						//break;
					}
					ImGui::PopID();

					ImGui::PushID(i+1);
					if (ImGui::BeginPopup(""))
					{
						ImGui::Text("Neurons Data");
						ImGui::Separator();
						if (ImGui::Selectable("Activate Layer"))
						{
							HermesNetwork::triggerLayer(L);
						}
						if (ImGui::Selectable("Backpropogate Error"))
						{
							HermesNetwork::backPropogateError(L);
						}
						if (ImGui::Selectable("Train Layer"))
						{
							HermesNetwork::trainLayer(L, &LR);
						}
						/*if (NEURONS_DATA == NULL)
							NEURONS_DATA = HermesNetwork::getLayerNeuronsData(NN, i+1);*/
						HermesNetwork::fetchLayerNeuronsData(L);
						ImGui::BeginChild("Neurons Data", { 300,40 }, true, ImGuiWindowFlags_HorizontalScrollbar);
						for (int j = 0; j < hiddenLayers[i]; j++)
						{
							ImGui::TextColored({ 1,0,0,1 }, "%f ", L->data[j]);
							ImGui::SameLine();
						}
						ImGui::EndChild();
						ImGui::EndPopup();
						//ImGui::PopID();
						//break;
					}
					ImGui::PopID();
				}

				if (ImGui::BeginPopup("1p"))
				{
					L = NN->outputLayer;
					ImGui::Text("Weights to Output Layer");
					ImGui::Separator();
					if (ImGui::Selectable("Re-Initialize Weights"))
					{						
						glBindImageTexture(0, L->WeightsTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
						glUseProgram(HermesNetwork::WeightInit);

						glUniform1i(HermesNetwork::WINT_unifm_seed, rand());
						glDispatchCompute(L->no_weight, 1, 1);
						glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
					}
					/*if (NEURONS_DATA == NULL)
						NEURONS_DATA = HermesNetwork::getWeights_Bias(NN, NN->no_layers - 1);*/
					HermesNetwork::fetchLayerWeights_Bias(L);

					{
						ImGui::BeginChild("Neurons Data", { 300,40 }, true, ImGuiWindowFlags_HorizontalScrollbar);
						for (int i = 0; i < NN->outputLayer->no_weight; i++)
						{
							ImGui::TextColored({ 1,0,0,1 }, "%f ", L->weights[i]);							
							ImGui::SameLine();
						}

						ImGui::EndChild();
					}
					ImGui::EndPopup();
				}
				else if (ImGui::BeginPopup("outL_DT"))
				{
					L = NN->outputLayer;
					ImGui::Text("Output Layer");
					ImGui::Separator();
					if (ImGui::Selectable("Activate Layer"))
					{						
						HermesNetwork::triggerLayer(NN->outputLayer);
					}
					if (ImGui::Selectable("Generate Error"))
					{
						HermesNetwork::calcError(NN->outputLayer, OUTPUTdata);
					}
					if (ImGui::Selectable("Train Layer"))
					{
						HermesNetwork::trainLayer(NN->outputLayer, &LR);
					}
					/*if (NEURONS_DATA == NULL)
						NEURONS_DATA = HermesNetwork::getLayerNeuronsData(NN,NN->no_layers-1);*/
					HermesNetwork::fetchLayerNeuronsData(L);
					//else
					{
						ImGui::BeginChild("Neurons Data", { 300,40 }, true, ImGuiWindowFlags_HorizontalScrollbar);
						for (int i = 0; i < Input_OutputLayer[1]; i++)
						{
							ImGui::TextColored({ 1,0,0,1 }, "%f ", L->data[i]);
							ImGui::SameLine();
						}

						ImGui::EndChild();
					}
					ImGui::EndPopup();
				}
				else
				{
					/*if (NEURONS_DATA != NULL)
						NEURONS_DATA = NULL;		*/
					
				}
			}
		}
		ImGui::End();

		DrawPingPong();	
		DrawLogicGateTrainer();
		
		ImGui::EndCleanWindow(window);
	}
	ImGui::PopStyleVar(2);
	ImGui::terminateImGui();
	ImGui::terminateGLFW(window);
	return 0;
}


void CreateNeuralNetwork() {
	INPUTdata = new float[Input_OutputLayer[0]]();
	OUTPUTdata = new float[Input_OutputLayer[1]]();
	CTRLPNL = true;
	TexLYRparts = WinPOS.y / ((hiddenLayers.size() + 2) * 2 - 1) - 20;

	GLint whichID, fbID;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &whichID);
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbID);

	NN = NetworkBuilder(Input_OutputLayer[0], hiddenLayers, Input_OutputLayer[1]);
	TriggerNetwork(NN);
}

void DeleteNeuralNetwork() {
	Input_OutputLayer[0] = 1;
	Input_OutputLayer[1] = 1;
	hiddenLayers.clear();
	hiddenLayerSize = 0;
	CTRLPNL = false;
	delete NN;
	NN = NULL;
}



void Theme1()
{	
	//ImGui::GetIO().Fonts->AddFontFromFileTTF("../data/Fonts/Ruda-Bold.ttf", 15.0f, &config);
	ImGui::GetStyle().FrameRounding = 5.0f;
	ImGui::GetStyle().GrabRounding = 4.0f;
	
	ImVec4* colors = ImGui::GetStyle().Colors;
	colors[ImGuiCol_Text] = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.36f, 0.42f, 0.47f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.15f, 0.18f, 0.22f, 1.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
	colors[ImGuiCol_Border] = ImVec4(0.08f, 0.10f, 0.12f, 1.00f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.12f, 0.20f, 0.28f, 1.00f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.09f, 0.12f, 0.14f, 1.00f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.09f, 0.12f, 0.14f, 0.65f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.08f, 0.10f, 0.12f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.15f, 0.18f, 0.22f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.39f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.18f, 0.22f, 0.25f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.09f, 0.21f, 0.31f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.28f, 0.56f, 1.00f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.28f, 0.56f, 1.00f, 1.00f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.37f, 0.61f, 1.00f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.28f, 0.56f, 1.00f, 1.00f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.20f, 0.25f, 0.29f, 0.55f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_Separator] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.10f, 0.40f, 0.75f, 0.78f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.10f, 0.40f, 0.75f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
	colors[ImGuiCol_Tab] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
	colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.25f, 0.29f, 1.00f);
	colors[ImGuiCol_TabUnfocused] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
	colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.11f, 0.15f, 0.17f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
	colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
	colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
	colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
}






void DrawPingPong() {
	if(PingPong::Enable) {
		if(ImGui::Begin("Ping Pong EvE", &PingPong::Enable, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar)) {
			ImGui::SetWindowSize({ 940, 600 });
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, { 2,1 });
			float ZZ[] = { 0,0, 1 }, YY[] = { 0,0.3 };
			//Neural Net1 working//////////////
			PingPong::input[0] = PingPong::ballpos.y / 550;
			PingPong::input[1] = PingPong::plr1Pos / 550;
			PingPong::input[2] = PingPong::ballangle;
			SendInputs(PingPong::N1, PingPong::input);
			TriggerNetwork(PingPong::N1);
			//PingPong::out1 = 
			FetchOutputLayerData(PingPong::N1);
			/////////////////////////////////////////
			//Neural Net2 working//////////////
			PingPong::input[1] = PingPong::plr2Pos / 550;
			SendInputs(PingPong::N2, PingPong::input);
			TriggerNetwork(PingPong::N2);
			//PingPong::out2 = GetOutputLayerData(PingPong::N2);
			FetchOutputLayerData(PingPong::N2);
			/////////////////////////////////////////


			//input
			if (PingPong::N1->Out[0] > PingPong::N1->Out[1])
				PingPong::plr1Pos += -6.2 * PingPong::speed;
			else
				PingPong::plr1Pos += 6.2 * PingPong::speed;

			if (PingPong::N2->Out[0] > PingPong::N2->Out[1])
				PingPong::plr2Pos += -6.2 * PingPong::speed;
			else
				PingPong::plr2Pos += 6.2 * PingPong::speed;

			//calc physics
			if (PingPong::plr1Pos < 30)
				PingPong::plr1Pos = 30;
			if (PingPong::plr1Pos > 380)
				PingPong::plr1Pos = 380;

			if (PingPong::plr2Pos < 30)
				PingPong::plr2Pos = 30;
			if (PingPong::plr2Pos > 380)
				PingPong::plr2Pos = 380;

			if (!PingPong::pause)
			{
				PingPong::ballpos.x += PingPong::ballspeed;
				PingPong::ballpos.y += PingPong::ballangle;

			}

			if (PingPong::ballpos.x < 40 && PingPong::ballpos.y > PingPong::plr1Pos && PingPong::ballpos.y < PingPong::plr1Pos + 200)
			{
				PingPong::ballspeed *= -1;
				PingPong::ballspeed += 0.8;
				PingPong::ballangle += (PingPong::plr1Pos + 100 - PingPong::ballpos.x) / 270;

				//Acknowledge
				if (PingPong::ballpos.y < PingPong::plr1Pos + 100)
				{
					PingPong::actualOUT[0] = 1;
					PingPong::actualOUT[1] = 0;
				}
				else
				{
					PingPong::actualOUT[0] = 0;
					PingPong::actualOUT[1] = 1;
				}
				if(PingPong::nn1Learn)
					TrainNetwork(PingPong::N1, PingPong::actualOUT, PingPong::ACK);
				//HermesNetwork::trainLayer(N1->outputLayer, actualOUT, &ACK);
			}
			if (PingPong::ballpos.x > 750 && PingPong::ballpos.y > PingPong::plr2Pos && PingPong::ballpos.y < PingPong::plr2Pos + 200)
			{
				PingPong::ballspeed *= -1;
				PingPong::ballspeed -= 0.8;
				PingPong::ballangle += (PingPong::plr2Pos + 100 - PingPong::ballpos.x) / 270;

				//Acknowledge
				if (PingPong::ballpos.y < PingPong::plr2Pos + 100)
				{
					PingPong::actualOUT[0] = 1;
					PingPong::actualOUT[1] = 0;
				}
				else
				{
					PingPong::actualOUT[0] = 0;
					PingPong::actualOUT[1] = 1;
				}
				if(PingPong::nn2Learn)
					TrainNetwork(PingPong::N2, PingPong::actualOUT, PingPong::ACK);
				//HermesNetwork::trainLayer(N2->outputLayer, actualOUT, &ACK);
			}
			if (PingPong::ballpos.y > 560 || PingPong::ballpos.y < 25)
				PingPong::ballangle *= -1;


				


			if (PingPong::ballpos.x > 800 || PingPong::ballpos.x < 0)
			{
				//train
				if (PingPong::ballpos.x < 0)
				{
					if (PingPong::ballpos.y < PingPong::plr1Pos + 100)
					{
						PingPong::actualOUT[0] = 1;
						PingPong::actualOUT[1] = 0;
					}
					else
					{
						PingPong::actualOUT[0] = 0;
						PingPong::actualOUT[1] = 1;
					}

					if(PingPong::nn1Learn)
						TrainNetwork(PingPong::N1, PingPong::actualOUT, PingPong::LR);
					//HermesNetwork::trainLayer(N1->outputLayer, actualOUT, &LR);
				}

				if (PingPong::ballpos.x > 800)
				{
					if (PingPong::ballpos.y < PingPong::plr2Pos + 100)
					{
						PingPong::actualOUT[0] = 1;
						PingPong::actualOUT[1] = 0;
					}
					else
					{
						PingPong::actualOUT[0] = 0;
						PingPong::actualOUT[1] = 1;
					}

					if(PingPong::nn2Learn)
						TrainNetwork(PingPong::N2, PingPong::actualOUT, PingPong::LR);				
					//PingPong::pause = true;
					//HermesNetwork::trainLayer(N2->outputLayer, actualOUT, &LR);
				}

				if (PingPong::ballspeed > 0)
					PingPong::scoreboard[0] += 1;
				else
					PingPong::scoreboard[1] += 1;
				PingPong::pause = true;
				PingPong::ballpos = { 400,300 };
				PingPong::ballangle = (int)((glfwGetTime()));
				PingPong::ballangle = (int)PingPong::ballangle % 20;
			}


			if (PingPong::ballspeed > 10.f)
				PingPong::ballspeed = 10.0f;

			//automatic continue
			if (PingPong::pause)
			{
				PingPong::pause = false;
				//PingPong::ballangle *= -1;

			}


			
			//draw
			ImGui::BeginChild("box", { 785,550 }, true);
			ImGui::EndChild();



			ImGui::PushStyleColor(ImGuiCol_ChildBg, { 0.6,0.6,0.6,1 });
			ImGui::PushStyleColor(ImGuiCol_Text, { 0,0,0,1 });
			ImGui::SetCursorPos({ 10,PingPong::plr1Pos });
			ImGui::BeginChild("a", { 30,200 }, true);
			/*ImGui::Text(" ^\n |");
			ImGui::Text("\n\n\n\n\n\n\n\n\n\ |\n V");*/
			ImGui::EndChild();

			//right plank
			ImGui::SetCursorPos({ 760,PingPong::plr2Pos });
			ImGui::BeginChild("b", { 30,200 }, true);
			/*ImGui::Text("Z");
			ImGui::Text("\n\n\n\n\n\n\n\n\n\n\nC");*/
			ImGui::EndChild();
			ImGui::PopStyleColor(2);




			//scoreboard
			ImGui::SetCursorPos({ 350, 220 });
			ImGui::BeginChild("scr", { 200,200 });

			ImGui::SetWindowFontScale(5);
			ImGui::Text("%d:%d", PingPong::scoreboard[0], PingPong::scoreboard[1]);
			ImGui::SetWindowFontScale(1);
			if (PingPong::pause) ImGui::Text("   Press Enter");
			ImGui::EndChild();

			//ball
			ImGui::PushStyleColor(ImGuiCol_ChildBg, { 0.6,0.9,0.6,1 });
			ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 100);
			ImGui::SetCursorPos(PingPong::ballpos);
			ImGui::BeginChild("o", { 20,20 }, true);
			ImGui::EndChild();
			ImGui::PopStyleVar();
			ImGui::PopStyleColor();

			// Network Images			
			ImGui::SetCursorPosX(820);
			ImGui::SetCursorPosY(30);
			ImGui::Text("Left Plank's NN");
			ImGui::SetCursorPosX(820);
			ImGui::Checkbox("Learn", &PingPong::nn1Learn);        
			ImGui::SameLine();
			ImGui::Button("Reset");
			ImGui::SetCursorPosX(820);
			ImGui::Image((ImTextureID)PingPong::N1->inputLayer->NeuronsTex, { 100,20 });
			PingPong::L = PingPong::N1->inputLayer->next;
			for (int i = 1; i < PingPong::N1->no_layers; i++)
			{
				ImGui::SetCursorPosX(820);
				ImGui::Image((ImTextureID)PingPong::L->WeightsTex, { 100,20 });
				ImGui::SetCursorPosX(820);
				ImGui::Image((ImTextureID)PingPong::L->NeuronsTex, { 100,20 });
				PingPong::L = PingPong::L->next;
			}


			ImGui::SetCursorPosX(820);
			ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 30);
			ImGui::Text("Right Plank's NN");
			ImGui::SetCursorPosX(820);
			ImGui::Checkbox("Learn ", &PingPong::nn2Learn);
			ImGui::SameLine();
			ImGui::Button("Reset");
			ImGui::SetCursorPosX(820);
			ImGui::Image((ImTextureID)PingPong::N2->inputLayer->NeuronsTex, { 100,20 });
			PingPong::L = PingPong::N2->inputLayer->next;
			for (int i = 1; i < PingPong::N2->no_layers; i++)
			{
				ImGui::SetCursorPosX(820);
				ImGui::Image((ImTextureID)PingPong::L->WeightsTex, { 100,20 });
				ImGui::SetCursorPosX(820);
				ImGui::Image((ImTextureID)PingPong::L->NeuronsTex, { 100,20 });
				PingPong::L = PingPong::L->next;
			}

			ImGui::PopStyleVar();
			ImGui::End();
		}

		if(!PingPong::Enable) {
			delete PingPong::N1;
			delete PingPong::N2;
			PingPong::N1 = NULL;
			PingPong::N2 = NULL;
		}
	}
}


void DrawLogicGateTrainer() {
	if(LogicGateTrainer::enable) {
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20,20));
		if(ImGui::Begin("Logic Gate Trainer", &LogicGateTrainer::enable, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar)) {
			if(!LogicGateTrainer::networkBuilt) {
				ImGui::Checkbox("AND", &LogicGateTrainer::andTrain);
				ImGui::SameLine();
				ImGui::Checkbox("OR",&LogicGateTrainer::orTrain);
				ImGui::SameLine();
				ImGui::Checkbox("XOR",&LogicGateTrainer::xorTrain);
				ImGui::SetNextItemWidth(120);

				int outputs = LogicGateTrainer::andTrain + LogicGateTrainer::orTrain + LogicGateTrainer::xorTrain;

				ImGui::Text("Hidden Layers");
				if(ImGui::InputInt("size", &LogicGateTrainer::hiddenLayerSize, 1, 1)) {
					if(LogicGateTrainer::hiddenLayerSize < 0)
						LogicGateTrainer::hiddenLayerSize = 0;
					if(LogicGateTrainer::hiddenLayerSize < LogicGateTrainer::hiddenLayers.size())
						LogicGateTrainer::hiddenLayers.pop_back();
					if(LogicGateTrainer::hiddenLayerSize > LogicGateTrainer::hiddenLayers.size())
						LogicGateTrainer::hiddenLayers.push_back(1);
				}
				if(ImGui::BeginChild("hl child", { 0,100 }, true)) {
					for (int i = 0; i < LogicGateTrainer::hiddenLayers.size(); i++) {
						ImGui::PushID(i);							
						ImGui::DragInt("", &LogicGateTrainer::hiddenLayers[i],1);
						ImGui::SameLine();
						ImGui::Text("%d",i+1);
						ImGui::PopID();
					}
					ImGui::EndChild();
				}

				if(outputs > 0 && ImGui::Button("Build Network")) {
					DeleteNeuralNetwork();
					LogicGateTrainer::networkBuilt = true;
					Input_OutputLayer[0] = 2;
					Input_OutputLayer[1] = outputs;
					hiddenLayers = LogicGateTrainer::hiddenLayers;
					CreateNeuralNetwork();
				}
			}
			else {
				if (ImGui::BeginTable("TT", 2 + Input_OutputLayer[1], ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
					ImGui::TableSetupColumn("A   ");
					ImGui::TableSetupColumn("B   ");
					if(LogicGateTrainer::andTrain)
						ImGui::TableSetupColumn("AND  ");
					if(LogicGateTrainer::orTrain)
						ImGui::TableSetupColumn("OR  ");
					if(LogicGateTrainer::xorTrain)
						ImGui::TableSetupColumn("XOR  ");

					ImGui::TableHeadersRow();
					
					for(int i = 0; i < 4; i++) {
						ImGui::TableNextRow();
						for(int j = 0; j < 2; j++) {
							ImGui::TableNextColumn();
							ImGui::Text("%d", LogicGateTrainer::TT[i][j]);
						}

						for(int k = 0; k < Input_OutputLayer[1]; k++) {
							ImGui::TableNextColumn();
							ImGui::Text("%f", LogicGateTrainer::Result[i][k]);
							ImU32 cell_bg_color;
							if(LogicGateTrainer::Result[i][k] > 0.5f)
								cell_bg_color = ImGui::GetColorU32(ImVec4(0.3f, 0.3f, 0.7f, 0.65f));
							else
								cell_bg_color = ImGui::GetColorU32(ImVec4(0.7f, 0.3f, 0.3f, 0.65f));
                        	ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, cell_bg_color);
						}
						
					}
					ImGui::EndTable();
				}

				ImGui::Spacing();
				ImGui::Button("Learn [click & hold]");
				if(ImGui::IsItemActive()) {
					INPUTdata[0] = LogicGateTrainer::TT[LogicGateTrainer::inpIndex][0];
					INPUTdata[1] = LogicGateTrainer::TT[LogicGateTrainer::inpIndex][1];
					SendInputs(NN, INPUTdata);
					TriggerNetwork(NN);
					FetchOutputLayerData(NN);

					for(int k = 0; k < Input_OutputLayer[1]; k++) {
						LogicGateTrainer::Result[LogicGateTrainer::inpIndex][k] = NN->Out[k];
					}

					int outIdx = 0;
					if(LogicGateTrainer::andTrain) {
						OUTPUTdata[outIdx] = LogicGateTrainer::AND[LogicGateTrainer::inpIndex];
						outIdx ++;
					}
					if(LogicGateTrainer::orTrain) {
						OUTPUTdata[outIdx] = LogicGateTrainer::OR[LogicGateTrainer::inpIndex];
						outIdx ++;
					}
					if(LogicGateTrainer::xorTrain) {
						OUTPUTdata[outIdx] = LogicGateTrainer::XOR[LogicGateTrainer::inpIndex];
						outIdx ++;						
					}

					TrainNetwork(NN, OUTPUTdata, LR);

					LogicGateTrainer::inpIndex ++;
					if(LogicGateTrainer::inpIndex > 3)
						LogicGateTrainer::inpIndex = 0;
				}

				if(ImGui::Button("Trigger")) {
					for(int i = 0; i < 4; i ++) {
						INPUTdata[0] = LogicGateTrainer::TT[i][0];
						INPUTdata[1] = LogicGateTrainer::TT[i][1];
						
						SendInputs(NN, INPUTdata);
						TriggerNetwork(NN);
						FetchOutputLayerData(NN);

						for(int k = 0; k < Input_OutputLayer[1]; k++) {
							LogicGateTrainer::Result[i][k] = NN->Out[k];
						}
					}
				}
				ImGui::SameLine();
				if(ImGui::Button("Clear")) {
					for(int k = 0; k < Input_OutputLayer[1]; k++) {
						for(int i = 0; i < 4; i++) {
							LogicGateTrainer::Result[i][k] = 0;
						}
					}
				}
			}
			ImGui::End();
		}

		if(!LogicGateTrainer::enable) {
			LogicGateTrainer::andTrain = false;
			LogicGateTrainer::orTrain = false;
			LogicGateTrainer::xorTrain = false;
			LogicGateTrainer::hiddenLayerSize = 0;
			LogicGateTrainer::hiddenLayers.clear();
			LogicGateTrainer::networkBuilt = false;
			LogicGateTrainer::inpIndex = 0;
			for(int i = 0; i < 4; i++) {
				for(int j = 0; j < 3; j ++) {
					LogicGateTrainer::Result[i][j] = 0;
				}
			}
		}

		ImGui::PopStyleVar();
	}
}
