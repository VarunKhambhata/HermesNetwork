include_directories(HermesNetworkInspector Dependencies/glfw/include Dependencies/CleanImGuiWin Dependencies/glew/include .)
link_directories(HermesNetworkInspector PUBLIC Dependencies/glew/lib/Release/x64 Dependencies/glfw/lib-mingw-w64)

add_subdirectory(Dependencies)

add_executable(HermesNetworkInspector main.cpp)
set_target_properties(HermesNetworkInspector PROPERTIES OUTPUT_NAME "HermesNetwork Inspector")


target_link_libraries(HermesNetworkInspector glew32s opengl32 glfw3 gdi32 Imm32 imgui -static-libstdc++)
#target_compile_definitions(HermesNetworkInspector PRIVATE GLEW_BUILD)