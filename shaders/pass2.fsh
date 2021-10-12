#version 330 core

in vec3 pix;
out vec4 fragColor;

uniform sampler2D texPass0;

void main() {
    fragColor = vec4(texture(texPass0, pix.xy * 0.5 + 0.5).rgb, 1.0);
}


