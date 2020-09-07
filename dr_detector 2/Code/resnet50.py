def create_attention_w_resnet50_model():
    in_lay = Input(t_x.shape[1:])
    
    base_pretrained_model = InceptionResNetV2(input_shape =  t_x.shape[1:], include_top = False, weights = 'imagenet')
    base_pretrained_model.trainable = False
    
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_features = base_pretrained_model(in_lay)
    
    bn_features = BatchNormalization()(pt_features)

    # here we do an attention mechanism to turn pixels in the GAP on an off

    attention_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
    attention_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attention_layer)
    attention_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attention_layer)
    attention_layer = Conv2D(1, kernel_size = (1,1), padding = 'valid', activation = 'sigmoid')(attention_layer)
    
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', activation = 'linear', use_bias = False, weights = [up_c2_w])
    up_c2.trainable = False
    
    attention_layer = up_c2(attention_layer)

    mask_features = multiply([attention_layer, bn_features])
    
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attention_layer)
    
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))
    output_layer = Dense(t_y.shape[-1], activation = 'softmax')(dr_steps)
    classifier = Model(inputs = [in_lay], outputs = [output_layer])
    
    
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', top_2_accuracy])
    print(classifier.summary())
    return classifier