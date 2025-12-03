CUDA_VISIBLE_DEVICES=1 python test_cogvideo.py \
--prompt="a handsome man in his mid-20s with chiseled 8-pack abs—each muscle defined, taut, and sculpted with subtle vascular details. He has a lean, athletic physique (broad shoulders, tapered waist), sun-kissed olive skin with a natural glow, and sharp facial features: deep almond eyes, high cheekbones, a straight nose, and a soft, confident smile. His dark, slightly messy hair falls loosely over his forehead, and he’s wearing minimal black athletic shorts, standing in soft golden-hour sunlight with warm, diffused lighting that accentuates his abdominal definition and body contours. The background is a sleek, modern rooftop with distant city skyline bokeh, wind gently rustling his hair and the hem of his shorts. He moves slowly—rotating his torso, stretching slightly, and glancing toward the camera with a relaxed, charismatic demeanor—capturing smooth, fluid motion with crisp detail on his muscles and skin texture. Cinematic depth of field, soft shadows, and natural color grading (warm oranges and soft blues) for a high-end, editorial aesthetic." \
--model_path="/NAS/fangjf/CogVideo/CogVideoX-5b" \
--eraser_path="./cogvideox5b_nudity_erasure" \
--eraser_rank=128 \
--num_frames=32 \
--generate_clean \
--output_path="./cog5b_nudity3" \
--seed=42