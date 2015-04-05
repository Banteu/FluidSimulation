#version 400

layout(location = 0) in vec3 vert;
uniform mat4 current_projection_matrix;
uniform mat4 current_modelview_matrix;
uniform vec3 camera_position;

out vec3 camSpacePos;
out float sphereRad;
void main()
{

	vec4 tmp = current_modelview_matrix * vec4(vert, 1.0);
	camSpacePos = tmp.xyz * (1.0 / tmp.w);
	tmp = current_projection_matrix * tmp;
	sphereRad = (1 - tmp.z / tmp.w) * 50;
	gl_PointSize = sphereRad;

	gl_Position = tmp;

}