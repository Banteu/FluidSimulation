#version 330

uniform samplerCube txSmp;
in vec3 position;
in vec3 normalDir;

layout(location = 0) out vec4 color;


void main(void)
{
//vec4 curPix = texture(txSmp, vec3(position.y, -position.zx));
	float curPix = texture(txSmp, position).x;
	color = vec4(curPix);
}
