# python make_animation_data.py

manim -pql make_animation.py LogPSplineHero

ffmpeg \
  -i /Users/avi/Documents/projects/LogPSplinePSD/docs/animation/media/videos/make_animation/480p15/LogPSplineHero.mp4 \
  -c:v libvpx-vp9 \
  -crf 32 \
  -b:v 0 \
  -an \
  logpspline-hero.webm


  