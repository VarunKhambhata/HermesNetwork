#pragma once

#include<CleanImGuiWin.h>
#include "HermesNetwork.h"

namespace PingPong {
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
	float LR = 1.8, ACK = 0.2;
	bool nn1Learn = true, nn2Learn= true;
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