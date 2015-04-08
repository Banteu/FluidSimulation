#version 400
layout(location = 0) out float color;
uniform mat4 current_projection_matrix;
uniform mat4 current_modelview_matrix;
uniform vec3 camera_position;
in vec3 camSpacePos;
uniform int drawingPass;

in float sphereRad;
float sphr = 0.004;
void main(void)
{
	
	vec3 ps = vec3(gl_PointCoord * 2 - vec2(1.0), 0);
	float rr = dot(ps, ps);
	if (rr > 1.0)
	{
		discard;
	}
	ps.z = -sqrt(1.0 - rr);
    if (drawingPass == 1){
	    vec4 frPos = vec4(camSpacePos + ps * sphr, 1.0);
	    frPos = current_projection_matrix * frPos;
	    float depth = frPos.z / frPos.w;
        color = depth;
    }
    else if(drawingPass == 2)
    {
        color = exp(-rr) * 0.5;
    }
}
