#version 400
layout(location = 0) out vec4 color;
layout(location = 2) out vec4 deepColor;

uniform sampler2D tSam;
uniform sampler2D deepSam;
uniform samplerCube envir;
in vec2 txr;

uniform mat4 current_projection_matrix;
uniform mat4 current_modelview_matrix;

uniform mat4 oldProjection_matrix;
uniform mat4 oldModelview_matrix;
uniform mat4 inverted_matrix;
uniform mat4 oldCamera_position;

uniform vec3 camera_position;

uniform float texSizeX;
uniform float texSizeY;


vec3 lightPosition = vec3(20, 30, 100);
float lightIntensity = 3;


vec3 getEyespacePos(vec2 texCoord, float depth)
{
    vec3 crd = vec3(texCoord, depth);
	vec4 nps = inverted_matrix * vec4(crd, 1.0);
	crd = nps.xyz * (1.0 / nps.w);
	return crd;
}


vec3 getNormal(float depth, vec2 texCoord)
{
	vec2 tNpos = txr;
	vec3 cntrl = getEyespacePos(tNpos, depth);
	tNpos = txr + vec2(texSizeX, 0); 
	vec3 ddx = getEyespacePos(tNpos, texture2D(tSam, tNpos)) - cntrl;
	tNpos = txr + vec2(-texSizeX, 0);
	vec3 ddx2 = cntrl - getEyespacePos(tNpos, texture2D(tSam, tNpos));
	if (abs(ddx.z) > abs(ddx2.z))
	{
		ddx = ddx2;
	}

	tNpos = txr + vec2(0, texSizeY); 
	vec3 ddy = getEyespacePos(tNpos, texture2D(tSam, tNpos)) - cntrl;
	tNpos = txr + vec2(0, -texSizeY);
	vec3 ddy2 = cntrl - getEyespacePos(tNpos, texture2D(tSam, tNpos));
	if (abs(ddy.z) > abs(ddy2.z))
	{
		ddy = ddy2;
	}
	return normalize(cross(ddx, ddy));
}

float reflectance = 0.1;

vec3 difColor = vec3(0.3, 0.6, 1.0);

void main()
{
	vec3 nol;

    
	float deepTexture = texture2D(deepSam, txr).x;
	float depth = texture2D(tSam, txr).x;
     
	vec3 eyeSpacePs = getEyespacePos(txr, depth);
	gl_FragDepth = 1 - depth;
	if(depth <= 1e-20)
		return;

	nol = getNormal(depth,txr * 2 - vec2(1.0));
    
	vec3 lightVector = normalize(lightPosition - eyeSpacePs);
	vec3 viewVector = normalize(-eyeSpacePs);
    vec3 reflRay = reflect(viewVector, nol);
    vec3 refrRay = refract(viewVector, nol, 1.4);
    float reflBright = texture(envir, reflRay).x;
    float refrBright = texture(envir, refrRay).x;

	float alpha = dot(nol, lightVector) * 0.5 + 0.2;
	
	float vl = 1 - dot(nol, viewVector);
	float pvl = vl * vl;
	pvl = pvl * pvl * vl * (1 - reflectance) + reflectance;
	
    //vec3 outColor = difColor * alpha + reflBright * pvl;

	color = vec4(alpha + pvl * reflBright);
    
    //color = vec4(deepTexture);
	//deepColor = vec4(0, 1, 0.3, 1);
//	color = vec4(nol * 0.5 + 0.5, 1);
}