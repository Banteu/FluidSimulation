#version 400

layout(location = 1) in vec3 vert;
layout(location = 2) in vec2 tex;
uniform mat4 current_projection_matrix;
uniform mat4 current_modelview_matrix;
uniform vec3 camera_position;

out vec2 txr;

void main()
{
	txr = tex;
	gl_Position = current_projection_matrix * current_modelview_matrix * vec4(vert, 1.0);
}