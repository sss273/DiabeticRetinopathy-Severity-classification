def augmentor(out_size,intermediate_size = (640, 640),intermediate_trans = 'crop',batch_size = 16,horizontal_flip = True, vertical_flip = False, random_brightness = True, random_contrast = True,
				 random_saturation = True,random_hue = True,color_mode = 'rgb',preproc_func = preprocess_input, min_crop_percent = 0.001,max_crop_percent = 0.005,crop_probability = 0.5,
				rotation_range = 10):
	
	load_operations = load_images(out_size = intermediate_size, 
							   horizontal_flip=horizontal_flip, 
							   vertical_flip=vertical_flip, 
							   random_brightness = random_brightness,
							   random_contrast = random_contrast,
							   random_saturation = random_saturation,
							   random_hue = random_hue,
							   color_mode = color_mode,
							   preproc_func = preproc_func,
							   on_batch=False)
	def batch_operations(X, y):
		batch_size = tf.shape(X)[0]
		with tf.name_scope('transformation'):
			transforms = []
			identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
			if rotation_range > 0:
				angle_rad = rotation_range / 180 * np.pi
				angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
				transforms += [tf.contrib.image.angles_to_projective_transforms(angles, intermediate_size[0], intermediate_size[1])]
			if crop_probability > 0:
				crop_pct = tf.random_uniform([batch_size], min_crop_percent, max_crop_percent)
				left = tf.random_uniform([batch_size], 0, intermediate_size[0] * (1.0 - crop_pct))
				top = tf.random_uniform([batch_size], 0, intermediate_size[1] * (1.0 - crop_pct))
				crop_transform = tf.stack([
					  crop_pct,
					  tf.zeros([batch_size]), top,
					  tf.zeros([batch_size]), crop_pct, left,
					  tf.zeros([batch_size]),
					  tf.zeros([batch_size])
				  ], 1)
				coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), crop_probability)
				transforms += [tf.where(coin, crop_transform, tf.tile(tf.expand_dims(identity, 0), [batch_size, 1]))]
			if len(transforms)>0:
				X = tf.contrib.image.transform(X,tf.contrib.image.compose_transforms(*transforms),interpolation='BILINEAR') # or 'NEAREST'
			if intermediate_trans=='scale':
				X = tf.image.resize_images(X, out_size)
			elif intermediate_trans=='crop':
				X = tf.image.resize_image_with_crop_or_pad(X, out_size[0], out_size[1])
			else:
				raise ValueError('Invalid Operation {}'.format(intermediate_trans))
			return X, y
	def _create_pipeline(in_ds):
		batch_ds = in_ds.map(load_operations, num_parallel_calls=4).batch(batch_size)
		return batch_ds.map(batch_operations)
	return _create_pipeline

def flow_df(idg, in_df, path_col,y_col, shuffle = True, color_mode = 'rgb'):
    files_ds = tf.data.Dataset.from_tensor_slices((in_df[path_col].values, np.stack(in_df[y_col].values,0)))
    in_len = in_df[path_col].values.shape[0]
    while True:
        if shuffle:
            files_ds = files_ds.shuffle(in_len) # shuffle the whole dataset
        
        next_batch = idg(files_ds).repeat().make_one_shot_iterator().get_next()
        
        for i in range(max(in_len//32,1)):
            yield K.get_session().run(next_batch)