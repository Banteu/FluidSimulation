#version 330

layout(location = 0) in vec3 vert;
layout(location = 1) in vec3 norm;

out vec3 position;
out vec3 normalDir;

uniform mat4 current_projection_matrix;
uniform mat4 current_modelview_matrix;



void main()
{
	position = vert;
	normalDir = norm;
	gl_Position = current_projection_matrix * current_modelview_matrix * vec4(vert, 1);
}