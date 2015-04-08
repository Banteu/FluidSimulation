#version 400
layout(location = 0) out float color;
layout(location = 1) out float deepColor;

uniform sampler2D tSam;
uniform sampler2D deepSam;


in vec2 txr;
uniform float texSizeX;
uniform float texSizeY;

float rad = 14;
float scale = 0.08;
float intensityScale = 10.00;

void main()
{
	float depth = texture2D(tSam, txr).x;
	//float deep = texture2D(deepSam, txr).x;
	if(depth < 1e-20)
		return;
	float finDepth = 0;
    float finDeep = 0;
    float deep2;


	float weight = 0;
    // float weight2  = 0;

	for(float y = -rad; y <= rad; ++y)
	{
	    for (float x = -rad; x <= rad; ++x)
	    {
		    float dst = abs(x) + abs(y);
		    float r = dst * scale;
		    float exp1 = exp(-r * r);
		    float depth2 = texture2D(tSam, txr + vec2(x * texSizeX, y * texSizeY)).x;
            
		    float r2 = (depth2 - depth) * intensityScale;
		    float ex2 = exp(-r2 * r2);
		    finDepth += depth2 * exp1 * ex2;
            
		    weight += exp1 * ex2;
            // float deep2 = texture2D(deepSam, txr + vec2(x * texSizeX, y * texSizeY)).x;
            // finDeep += deep2 * exp1;
            //weight2 += exp1;
	    }
	}
	if (weight > 0)
	{
		finDepth = finDepth / weight;        
	}
     //if(weight2 > 0)
     // {
     //    finDeep = finDeep / weight2;
     // }
	color = finDepth;
}