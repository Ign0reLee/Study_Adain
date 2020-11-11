import tensorflow as tf
import tensorflow.keras as k


from tensorflow.keras.layers import  *
from tensorflow.keras.models import Model




class AdaIn_Transfer(Model):
    
    def __init__(self):
        super(AdaIn_Transfer, self).__init__(name = "AdaIn_Transfer")
        
        self.relu = ReLU()
        self.conv4_1 = Conv2D(filters = 256, kernel_size = (3,3), strides=(1, 1), padding='SAME')
        
        self.up1 = UpSampling2D(size=(2, 2))        
        self.conv3_4 = Conv2D(filters = 256, kernel_size = (3,3), strides=(1, 1), padding='SAME')
        self.conv3_3 = Conv2D(filters = 256, kernel_size = (3,3), strides=(1, 1), padding='SAME')
        self.conv3_2 = Conv2D(filters = 256, kernel_size = (3,3), strides=(1, 1), padding='SAME')
        self.conv3_1 = Conv2D(filters = 128, kernel_size = (3,3), strides=(1, 1), padding='SAME')
        
        self.up2 = UpSampling2D(size=(2, 2))        
        self.conv2_2 = Conv2D(filters = 128, kernel_size = (3,3), strides=(1, 1), padding='SAME')
        self.conv2_1 = Conv2D(filters = 64, kernel_size = (3,3), strides=(1, 1), padding='SAME')
        
        self.up3 = UpSampling2D(size=(2, 2))
        self.conv1_2 = Conv2D(filters = 64, kernel_size = (3,3), strides=(1, 1), padding='SAME')
        self.conv1_1 = Conv2D(filters = 3, kernel_size = (3,3), strides=(1, 1), padding='SAME')
       
    
        
    def call(self, content_inputs, style_inputs):
                
        self.out = self.AdaIN(content_inputs, style_inputs)
        
        h = self.relu(self.conv4_1(self.out))
        
        h = self.up1(h)        
        h = self.relu(self.conv3_4(h))
        h = self.relu(self.conv3_3(h))
        h = self.relu(self.conv3_2(h))
        h = self.relu(self.conv3_1(h))
        
        h = self.up2(h)        
        h = self.relu(self.conv2_2(h))
        h = self.relu(self.conv2_1(h))
        
        h = self.up3(h)
        h = self.relu(self.conv1_2(h))        
        h = self.relu(self.conv1_1(h))
        
        return h
        
    def AdaIN(self, content, style, eps=1e-5):
        
        content_mean, content_var = tf.nn.moments(content, axes =[1,2], keepdims = True)
        style_mean, style_var   = tf.nn.moments(style, axes =[1,2], keepdims=True)
        
        content_std =  tf.sqrt(content_var + eps)
        style_std   =  tf.sqrt(style_var + eps)
        norm = tf.math.divide((content-content_mean),content_std)
        
        return tf.math.add(tf.math.multiply(norm, style_std), style_mean)
    
    def Content_Loss(self, output, style):
        return tf.reduce_mean(tf.math.square(output-style))
    
    def Style_Loss(self, output, style, eps = 1e-5):
    
        output_mean, output_var = tf.nn.moments(output, axes = [1,2], keepdims=True)
        style_mean,  style_var  = tf.nn.moments(style,  axes = [1,2], keepdims=True )
        output_std, style_std   = tf.math.sqrt(output_var + eps), tf.math.sqrt(style_var + eps)

        mean_loss = tf.reduce_sum(tf.math.squared_difference(output_mean, style_mean))
        std_loss  = tf.reduce_sum(tf.math.squared_difference(output_std,  style_std))

        n = tf.cast(tf.shape(output)[0],dtype=tf.float32)
        mean_loss /= n
        std_loss  /= n

        #return tf.reduce_mean(tf.math.square(output_mean-style_mean)) + tf.reduce_mean(tf.math.square(output_std, style_std))
        return mean_loss + std_loss
    
    def AdaIN_Loss(self, g_features,c_features, s_features, lam = 10.0):
    
        loss_c = self.Content_Loss(g_features[-1], self.out)
        loss_s = self.Style_Loss(g_features[0], s_features[0])
        for i in range(1,4):
            loss_s += self.Style_Loss(g_features[i], s_features[i])

        return loss_c + (lam * loss_s)