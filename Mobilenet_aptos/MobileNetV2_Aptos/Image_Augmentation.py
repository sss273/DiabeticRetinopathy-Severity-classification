import Augmentor

p = Augmentor.Pipeline("/media/sandeep/New Volume/Angel Billy/Final dataset/4")

p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.2)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=16)
p.rotate(probability=1, max_left_rotation=45, max_right_rotation=45)
p.sample(1000)