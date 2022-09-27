import tensorflow as tf
import keras

class InpaintContextAttentionUnit5edge(keras.layers.Layer):
    def __init__(self, n=8, fil=16, **kwargs):
        super(InpaintContextAttentionUnit5edge,self).__init__(**kwargs)
        #self.conv2d = tf.keras.layers.Conv2D(filters=fil,kernel_size=[3,3],padding="valid",activation="relu")
        self.n = n
        self.fil = fil
        #filters need to be set as the input channels dim.

    def build(self, input_shape):
        self.conv2d = tf.keras.layers.Conv2D(filters=self.fil,kernel_size=[5,5],padding="valid",activation="relu")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n':self.n,
            'fil': self.fil})
        return config

    def call(self, feature_map, training=None):
        #print("Im in the call block!")

        def inpaint_rows_5_5(inputs):
            """
            The function for mimicking inpainting and then stacking t,he individual rows.
            """
            input_b = inputs.shape[0]
            input_h = inputs.shape[1]
            input_w = inputs.shape[2]
            input_c = inputs.shape[3]
            
            inpainted_tensors = []
            #adding padding to the tensor on all four sides.
            paddings = [[0,0],[2,2],[2,2],[0,0]]
            feature_map_padded = tf.pad(inputs,paddings=paddings,mode="CONSTANT")
            
            
            for row in range(feature_map_padded.shape[1]-4):
                tensor_1 = feature_map_padded[:,row:row+1,:,:]
                tensor_2 = tf.zeros(shape=[tf.shape(inputs)[0],1, input_w+4, input_c])
                tensor_mid = tf.zeros(shape=[tf.shape(inputs)[0],1, input_w+4, input_c])
                tensor_3 = tf.zeros(shape=[tf.shape(inputs)[0],1, input_w+4, input_c])
                tensor_4 = feature_map_padded[:,row+4:row+5,:,:]
                

                sub_tensor = tf.concat([tensor_1, tensor_2, tensor_mid, tensor_3, tensor_4],axis=1)
                #print("Top+Mid+Bottom Concat Shape ---->", sub_tensor.shape)
                sub_tensorCONV = self.conv2d(sub_tensor)
                #print("Applying Conv2D the shape becomes ----->(This is passed to inpainted_tensor list)", sub_tensorCONV.shape)
                inpainted_tensors.append(sub_tensorCONV)
            #print("Inpainted_Tensor length",len(inpainted_tensors))
            res = tf.concat(inpainted_tensors,axis=1)
            #print("Final Row OP out Dim -----> ####", res.shape)
            return res
        
        def inpaint_cols_5_5(inputs):
            #print("$$$$$Feature recd. by Col Op----->", inputs.shape)
            transposed_input = tf.transpose(inputs,[0,2,1,3])
            #print("$$$$Transposed Feature shape --->", transposed_input.shape)
            inpainted_transposed_input = inpaint_rows_5_5(transposed_input)
            #print("$$$$$Shape after inpainting---->", inpainted_transposed_input.shape)
            return tf.transpose(inpainted_transposed_input,[0,2,1,3])      


        h = feature_map.shape[1]
        w = feature_map.shape[2]
        n = self.n
        #print("Input Feature Shape --->", feature_map.shape)
        #Average Pooling the Input Feature.
        out = keras.layers.AveragePooling2D(pool_size=(h/n,2), strides=(h/n,2),padding="SAME")(feature_map)
        #print("Shape After Average Pooling---->", out.shape)
 
        #Applying Conv-Inpainting on the input features.
        #print("----FEATURE RECD. BY ROW OP. ---> ", out.shape)
        row_op = inpaint_rows_5_5(out)
        #print("----FEATURE OUTPUT BY COL OP. ---> ", row_op.shape)
        #print("row_op complete")
        col_op = inpaint_cols_5_5(out)
        #print("----FEATURE OUTPUT BY ROW OP. ---> ", col_op.shape)
        #print("col_op complete")
        #Upsampling the inpainted feature map
        #print("=====================================")

        row_op_upsampled = tf.image.resize(row_op,[h, w])
        col_op_upsampled = tf.image.resize(col_op,[h, w])
        #print("rowOP Upsample shape -->", row_op_upsampled.shape)
        #print("colOP Upsample shape -->", col_op_upsampled.shape)
        #Stacking the original Feature map and the Inpainted feature maps respectively.
        stacked_op = tf.concat([row_op_upsampled,col_op_upsampled],axis=3)
        #print("Stacked Tensor Shape ---->", stacked_op.shape)
        stacked_orig = tf.concat([feature_map, feature_map], axis=3)
        #print("Stack OG Shape---->", stacked_orig.shape)

        #Subtracting the inpainted features from the original feature map.
        diff_feature = tf.math.subtract(stacked_orig,stacked_op)
        res = tf.concat([feature_map,diff_feature],axis=3)
        #print("FINE AFTER DIFF....")
        #Applying CONV to match the shapes of the processed feature map to the input feature.
        #final = self.conv2d(diff_feature)
        #print("Reached Final")
        return res


#inputs = tf.keras.Input(shape=(8,16,32),batch_size=4)
#outputs = InpaintContextAttentionUnit5edge(fil=inputs.shape[3],n=2)(inputs)
#outputs = tf.keras.layers.Conv2D(filters = outputs.shape[3]/3,kernel_size=1, activation='relu', padding='SAME',data_format='channels_last')(outputs)
#model = tf.keras.Model(inputs=inputs, outputs=outputs, name='test')
#model.compile(loss=tf.keras.losses.MeanSquaredError())
#model.summary()
#
