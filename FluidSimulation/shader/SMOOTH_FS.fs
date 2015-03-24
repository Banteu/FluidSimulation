#version 400
layout(location = 0) out float color;
layout(location = 1) out float deepColor;

uniform sampler2D tSam;
uniform sampler2D deepSam;


in vec2 txr;
uniform float texSizeX;
uniform float texSizeY;

//float rad = 10;

float rad = 10;
float scale = 0.1;
float intensityScale = 20.00;

void main()
{
	float depth = texture2D(tSam, txr).x;
	float deep = texture2D(deepSam, txr).x;
	if(depth < 0.00001)
		return;
	float finDepth = 0;
	float weight = 0;
	for(float y = -rad; y <= rad; ++y)
	{
	for (float x = -rad; x <= rad; ++x)
	{
		float dst = sqrt(x * x + y * y);
		float r = dst * scale;
		float exp1 = exp(-r * r);
		float depth2 = texture2D(tSam, txr + vec2(x * texSizeX, y * texSizeY)).x;
		float r2 = (depth2 - depth) * intensityScale;
		float ex2 = exp(-r2 * r2);
		finDepth += depth2 * exp1 * ex2;
		weight += exp1 * ex2;
	}
	}

	if (weight > 0)
	{
		finDepth = finDepth / weight;
	}



	color = finDepth;
    deepColor = deep;
}