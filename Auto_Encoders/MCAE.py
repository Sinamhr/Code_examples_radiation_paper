import os
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda, \
        Conv3D, MaxPool3D, Conv3DTranspose, UpSampling3D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import numpy as np
import tensorflow as tf


tf.compat.v1.disable_eager_execution()


class MCAE:
    """
    MCAE represents a Deep Convolutional autoencoder architecture
    with mirrored encoder and decoder components and constrained latent space.
    """

    def __init__(self,
                 input_shape,
                 latent_space_dim):
        self.input_shape = input_shape # [82, 82, 1]
        #self.conv_filters = conv_filters # [2, 4, 8]
        #self.conv_kernels = conv_kernels # [(1 , 3, 3), (1 , 3, 3), (1 , 3, 3)]
        #self.Maxpooling = conv_strides # [1, 2, 2]
        #self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2
        self.reconstruction_loss_weight = 40

        self.encoder = None
        self.decoder = None
        self.model = None

        #self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss,
                           metrics=[self._calculate_reconstruction_loss,
                                    self._calculate_P_loss1,self._calculate_P_loss2, self._calculate_P_loss3, self._calculate_P_loss4, self._calculate_P_loss5,
                                    self._calculate_P_loss6,self._calculate_P_loss7, self._calculate_P_loss8, self._calculate_P_loss9, self._calculate_P_loss10,
                                    self._calculate_P_loss11,self._calculate_P_loss12, self._calculate_P_loss13, self._calculate_P_loss14, self._calculate_P_loss15,
                                    self._calculate_P_loss16,self._calculate_P_loss17, self._calculate_P_loss18, self._calculate_P_loss19, self._calculate_P_loss20,
                                    self._calculate_P_loss21,self._calculate_P_loss22, self._calculate_P_loss23, self._calculate_P_loss24, self._calculate_P_loss25,
                                    self._calculate_P_loss26,self._calculate_P_loss27, self._calculate_P_loss28, self._calculate_P_loss29, self._calculate_P_loss30,
                                    self._calculate_P_loss31,self._calculate_P_loss32, self._calculate_P_loss33, self._calculate_P_loss34, self._calculate_P_loss35,
                                    self._calculate_P_loss36,self._calculate_P_loss37, self._calculate_P_loss38, self._calculate_P_loss39, self._calculate_P_loss40])

    def train(self, x_train, x_test, batch_size, num_epochs, early_stop):
        Early_stop = EarlyStopping(monitor='val__calculate_reconstruction_loss', patience=early_stop, verbose=1)
        log_csv = CSVLogger('103_4_my_logs.csv', separator=',', append=True)
        callbacks_list = [Early_stop, log_csv]
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True,
                       validation_data=(x_test, x_test),
                       callbacks=callbacks_list)

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters_103_4.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = MCAE(*parameters)
        weights_path = os.path.join(save_folder, "weights_103_4.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        P_loss1 = self._calculate_P_loss1(y_target, y_predicted)
        P_loss2 = self._calculate_P_loss2(y_target, y_predicted)
        P_loss3 = self._calculate_P_loss3(y_target, y_predicted)
        P_loss4 = self._calculate_P_loss4(y_target, y_predicted)
        P_loss5 = self._calculate_P_loss5(y_target, y_predicted)
        P_loss6 = self._calculate_P_loss6(y_target, y_predicted)
        P_loss7 = self._calculate_P_loss7(y_target, y_predicted)
        P_loss8 = self._calculate_P_loss8(y_target, y_predicted)
        P_loss9 = self._calculate_P_loss9(y_target, y_predicted)
        P_loss10 = self._calculate_P_loss10(y_target, y_predicted)
        P_loss11 = self._calculate_P_loss11(y_target, y_predicted)
        P_loss12 = self._calculate_P_loss12(y_target, y_predicted)
        P_loss13 = self._calculate_P_loss13(y_target, y_predicted)
        P_loss14 = self._calculate_P_loss14(y_target, y_predicted)
        P_loss15 = self._calculate_P_loss15(y_target, y_predicted)
        P_loss16 = self._calculate_P_loss16(y_target, y_predicted)
        P_loss17 = self._calculate_P_loss17(y_target, y_predicted)
        P_loss18 = self._calculate_P_loss18(y_target, y_predicted)
        P_loss19 = self._calculate_P_loss19(y_target, y_predicted)
        P_loss20 = self._calculate_P_loss20(y_target, y_predicted)
        P_loss21 = self._calculate_P_loss21(y_target, y_predicted)
        P_loss22 = self._calculate_P_loss22(y_target, y_predicted)
        P_loss23 = self._calculate_P_loss23(y_target, y_predicted)
        P_loss24 = self._calculate_P_loss24(y_target, y_predicted)
        P_loss25 = self._calculate_P_loss25(y_target, y_predicted)
        P_loss26 = self._calculate_P_loss26(y_target, y_predicted)
        P_loss27 = self._calculate_P_loss27(y_target, y_predicted)
        P_loss28 = self._calculate_P_loss28(y_target, y_predicted)
        P_loss29 = self._calculate_P_loss29(y_target, y_predicted)
        P_loss30 = self._calculate_P_loss30(y_target, y_predicted)
        P_loss31 = self._calculate_P_loss31(y_target, y_predicted)
        P_loss32 = self._calculate_P_loss32(y_target, y_predicted)
        P_loss33 = self._calculate_P_loss33(y_target, y_predicted)
        P_loss34 = self._calculate_P_loss34(y_target, y_predicted)
        P_loss35 = self._calculate_P_loss35(y_target, y_predicted)
        P_loss36 = self._calculate_P_loss36(y_target, y_predicted)
        P_loss37 = self._calculate_P_loss37(y_target, y_predicted)
        P_loss38 = self._calculate_P_loss38(y_target, y_predicted)
        P_loss39 = self._calculate_P_loss39(y_target, y_predicted)
        P_loss40 = self._calculate_P_loss40(y_target, y_predicted)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss\
                                                         + P_loss1 + P_loss2 + P_loss3 + P_loss4 + P_loss5 + P_loss6 + P_loss7 + P_loss8 + P_loss9 + P_loss10\
                                                             + P_loss11 + P_loss12 + P_loss13 + P_loss14 + P_loss15 + P_loss16 + P_loss17 + P_loss18 + P_loss19 + P_loss20\
                                                                 + P_loss21 + P_loss22 + P_loss23 + P_loss24 + P_loss25 + P_loss26 + P_loss27 + P_loss28 + P_loss29 + P_loss30\
                                                                     + P_loss31 + P_loss32 + P_loss33 + P_loss34 + P_loss35 + P_loss36 + P_loss37 + P_loss38 + P_loss39 + P_loss40
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3, 4])
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.latent_space) -
                               K.exp(self.log_variance), axis=1)
        return kl_loss


    def _calculate_P_loss1(self, y_target, y_predicted):
        error = self.parameters_1 [:] - self.latent_space [:,0:1]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss2(self, y_target, y_predicted):
        error = self.parameters_2 [:] - self.latent_space [:,1:2]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss3(self, y_target, y_predicted):
        error = self.parameters_3 [:] - self.latent_space [:,2:3]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss4(self, y_target, y_predicted):
        error = self.parameters_4 [:] - self.latent_space [:,3:4]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss5(self, y_target, y_predicted):
        error = self.parameters_5 [:] - self.latent_space [:,4:5]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss6(self, y_target, y_predicted):
        error = self.parameters_6 [:] - self.latent_space [:,5:6]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss7(self, y_target, y_predicted):
        error = self.parameters_7 [:] - self.latent_space [:,6:7]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss8(self, y_target, y_predicted):
        error = self.parameters_8 [:] - self.latent_space [:,7:8]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss9(self, y_target, y_predicted):
        error = self.parameters_9 [:] - self.latent_space [:,8:9]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss
        
    def _calculate_P_loss10(self, y_target, y_predicted):
        error = self.parameters_10 [:] - self.latent_space [:,9:10]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss11(self, y_target, y_predicted):
        error = self.parameters_11 [:] - self.latent_space [:,10:11]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss12(self, y_target, y_predicted):
        error = self.parameters_12 [:] - self.latent_space [:,11:12]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss13(self, y_target, y_predicted):
        error = self.parameters_13 [:] - self.latent_space [:,12:13]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss14(self, y_target, y_predicted):
        error = self.parameters_14 [:] - self.latent_space [:,13:14]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss15(self, y_target, y_predicted):
        error = self.parameters_15 [:] - self.latent_space [:,14:15]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss16(self, y_target, y_predicted):
        error = self.parameters_16 [:] - self.latent_space [:,15:16]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss17(self, y_target, y_predicted):
        error = self.parameters_17 [:] - self.latent_space [:,16:17]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss18(self, y_target, y_predicted):
        error = self.parameters_18 [:] - self.latent_space [:,17:18]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss
        
    def _calculate_P_loss19(self, y_target, y_predicted):
        error = self.parameters_19 [:] - self.latent_space [:,18:19]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss20(self, y_target, y_predicted):
        error = self.parameters_20 [:] - self.latent_space [:,19:20]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss21(self, y_target, y_predicted):
        error = self.parameters_21 [:] - self.latent_space [:,20:21]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss22(self, y_target, y_predicted):
        error = self.parameters_22 [:] - self.latent_space [:,21:22]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss23(self, y_target, y_predicted):
        error = self.parameters_23 [:] - self.latent_space [:,22:23]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss24(self, y_target, y_predicted):
        error = self.parameters_24 [:] - self.latent_space [:,23:24]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss25(self, y_target, y_predicted):
        error = self.parameters_25 [:] - self.latent_space [:,24:25]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss26(self, y_target, y_predicted):
        error = self.parameters_26 [:] - self.latent_space [:,25:26]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss27(self, y_target, y_predicted):
        error = self.parameters_27 [:] - self.latent_space [:,26:27]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss28(self, y_target, y_predicted):
        error = self.parameters_28 [:] - self.latent_space [:,27:28]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss29(self, y_target, y_predicted):
        error = self.parameters_29 [:] - self.latent_space [:,28:29]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss30(self, y_target, y_predicted):
        error = self.parameters_30 [:] - self.latent_space [:,29:30]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss31(self, y_target, y_predicted):
        error = self.parameters_31 [:] - self.latent_space [:,30:31]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss32(self, y_target, y_predicted):
        error = self.parameters_32 [:] - self.latent_space [:,31:32]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss33(self, y_target, y_predicted):
        error = self.parameters_33 [:] - self.latent_space [:,32:33]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss34(self, y_target, y_predicted):
        error = self.parameters_34 [:] - self.latent_space [:,33:34]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss35(self, y_target, y_predicted):
        error = self.parameters_35 [:] - self.latent_space [:,34:35]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss36(self, y_target, y_predicted):
        error = self.parameters_36 [:] - self.latent_space [:,35:36]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss37(self, y_target, y_predicted):
        error = self.parameters_37 [:] - self.latent_space [:,36:37]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss38(self, y_target, y_predicted):
        error = self.parameters_38 [:] - self.latent_space [:,37:38]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss
        
    def _calculate_P_loss39(self, y_target, y_predicted):
        error = self.parameters_39 [:] - self.latent_space [:,38:39]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss

    def _calculate_P_loss40(self, y_target, y_predicted):
        error = self.parameters_40 [:] - self.latent_space [:,39:40]
        p_loss = K.mean(K.square(error), axis=[1])
        return p_loss


    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.latent_space_dim,
        ]
        save_path = os.path.join(save_folder, "parameters_103_4.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights_103_4.h5")
        self.model.save_weights(save_path)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output1 = self.decoder(self.encoder(model_input))
        model_output = tf.concat([ self.extra_data, model_output1], 2)
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = Input(shape=self.latent_space_dim, name="decoder_input")
        
        x = Dense(200, activation='relu')(decoder_input)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Reshape((self.conv_shape[1], self.conv_shape[2], self.conv_shape[3], self.conv_shape[4]))(x)

        x = Conv3DTranspose(filters=256, kernel_size=(1 , 3, 3), padding='same', activation='relu')(x)
        x = UpSampling3D(size=(1, 2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv3DTranspose(filters=128, kernel_size=(1 , 3, 3), padding='same', activation='relu')(x)
        x = UpSampling3D(size=(1, 2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv3DTranspose(filters=128, kernel_size=(1 , 3, 3), padding='valid', activation='relu')(x)
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv3DTranspose(filters=64, kernel_size=(1 , 3, 3), padding='same', activation='relu')(x)
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv3DTranspose(filters=32, kernel_size=(1 , 3, 3), padding='same', activation='relu')(x)
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv3DTranspose(filters=32, kernel_size=(1 , 3, 3), padding='valid', activation='relu')(x)
        x = Conv3DTranspose(filters=2, kernel_size=(3 , 3, 3), padding='same', activation='sigmoid')(x)  #name='decoder_output'
        
        self.decoder = Model(decoder_input, x, name='decoder')

    def _build_encoder(self):
        encoder_input = Input(shape=self.input_shape, name='encoder_input')
        
        def Spliter1(X):
            return X[:,:,1:,:,:]
        def Spliter2(X):
            return X[:,:,0:1,:,:]
        
        def Spliter_1(X):
            return X[:,0,0,0:1,0]

        def Spliter_2(X):
            return X[:,0,0,1:2,0]

        def Spliter_3(X):
            return X[:,0,0,2:3,0]

        def Spliter_4(X):
            return X[:,0,0,3:4,0]

        def Spliter_5(X):
            return X[:,0,0,4:5,0]

        def Spliter_6(X):
            return X[:,0,0,5:6,0]

        def Spliter_7(X):
            return X[:,0,0,6:7,0]

        def Spliter_8(X):
            return X[:,0,0,7:8,0]

        def Spliter_9(X):
            return X[:,0,0,8:9,0]

        def Spliter_10(X):
            return X[:,0,0,9:10,0]

        def Spliter_11(X):
            return X[:,0,0,10:11,0]

        def Spliter_12(X):
            return X[:,0,0,11:12,0]

        def Spliter_13(X):
            return X[:,0,0,12:13,0]

        def Spliter_14(X):
            return X[:,0,0,13:14,0]

        def Spliter_15(X):
            return X[:,0,0,14:15,0]

        def Spliter_16(X):
            return X[:,0,0,15:16,0]

        def Spliter_17(X):
            return X[:,0,0,16:17,0]

        def Spliter_18(X):
            return X[:,0,0,17:18,0]

        def Spliter_19(X):
            return X[:,0,0,18:19,0]

        def Spliter_20(X):
            return X[:,0,0,19:20,0]   

        def Spliter_21(X):
            return X[:,0,0,20:21,0]

        def Spliter_22(X):
            return X[:,0,0,21:22,0]

        def Spliter_23(X):
            return X[:,0,0,22:23,0]

        def Spliter_24(X):
            return X[:,0,0,23:24,0]

        def Spliter_25(X):
            return X[:,0,0,24:25,0]

        def Spliter_26(X):
            return X[:,0,0,25:26,0]

        def Spliter_27(X):
            return X[:,0,0,26:27,0]

        def Spliter_28(X):
            return X[:,0,0,27:28,0]

        def Spliter_29(X):
            return X[:,0,0,28:29,0]

        def Spliter_30(X):
            return X[:,0,0,29:30,0] 

        def Spliter_31(X):
            return X[:,0,0,30:31,0]

        def Spliter_32(X):
            return X[:,0,0,31:32,0]

        def Spliter_33(X):
            return X[:,0,0,32:33,0]

        def Spliter_34(X):
            return X[:,0,0,33:34,0]

        def Spliter_35(X):
            return X[:,0,0,34:35,0]

        def Spliter_36(X):
            return X[:,0,0,35:36,0]

        def Spliter_37(X):
            return X[:,0,0,36:37,0]

        def Spliter_38(X):
            return X[:,0,0,37:38,0]

        def Spliter_39(X):
            return X[:,0,0,38:39,0]

        def Spliter_40(X):
            return X[:,0,0,39:40,0]  
        
        self.parameters_1 = Lambda(Spliter_1)(encoder_input)
        self.parameters_2 = Lambda(Spliter_2)(encoder_input)
        self.parameters_3 = Lambda(Spliter_3)(encoder_input)
        self.parameters_4 = Lambda(Spliter_4)(encoder_input)
        self.parameters_5 = Lambda(Spliter_5)(encoder_input)
        self.parameters_6 = Lambda(Spliter_6)(encoder_input)
        self.parameters_7 = Lambda(Spliter_7)(encoder_input)
        self.parameters_8 = Lambda(Spliter_8)(encoder_input)
        self.parameters_9 = Lambda(Spliter_9)(encoder_input)
        self.parameters_10 = Lambda(Spliter_10)(encoder_input)
        self.parameters_11 = Lambda(Spliter_11)(encoder_input)
        self.parameters_12 = Lambda(Spliter_12)(encoder_input)
        self.parameters_13 = Lambda(Spliter_13)(encoder_input)
        self.parameters_14 = Lambda(Spliter_14)(encoder_input)
        self.parameters_15 = Lambda(Spliter_15)(encoder_input)
        self.parameters_16 = Lambda(Spliter_16)(encoder_input)
        self.parameters_17 = Lambda(Spliter_17)(encoder_input)
        self.parameters_18 = Lambda(Spliter_18)(encoder_input)
        self.parameters_19 = Lambda(Spliter_19)(encoder_input)
        self.parameters_20 = Lambda(Spliter_20)(encoder_input)
        self.parameters_21 = Lambda(Spliter_21)(encoder_input)
        self.parameters_22 = Lambda(Spliter_22)(encoder_input)
        self.parameters_23 = Lambda(Spliter_23)(encoder_input)
        self.parameters_24 = Lambda(Spliter_24)(encoder_input)
        self.parameters_25 = Lambda(Spliter_25)(encoder_input)
        self.parameters_26 = Lambda(Spliter_26)(encoder_input)
        self.parameters_27 = Lambda(Spliter_27)(encoder_input)
        self.parameters_28 = Lambda(Spliter_28)(encoder_input)
        self.parameters_29 = Lambda(Spliter_29)(encoder_input)
        self.parameters_30 = Lambda(Spliter_30)(encoder_input)
        self.parameters_31 = Lambda(Spliter_31)(encoder_input)
        self.parameters_32 = Lambda(Spliter_32)(encoder_input)
        self.parameters_33 = Lambda(Spliter_33)(encoder_input)
        self.parameters_34 = Lambda(Spliter_34)(encoder_input)
        self.parameters_35 = Lambda(Spliter_35)(encoder_input)
        self.parameters_36 = Lambda(Spliter_36)(encoder_input)
        self.parameters_37 = Lambda(Spliter_37)(encoder_input)
        self.parameters_38 = Lambda(Spliter_38)(encoder_input)
        self.parameters_39 = Lambda(Spliter_39)(encoder_input)
        self.parameters_40 = Lambda(Spliter_40)(encoder_input)
        
        self.extra_data = Lambda(Spliter2)(encoder_input)
        
        x = Lambda(Spliter1)(encoder_input)
        x = Conv3D(filters=32, kernel_size=(3 , 3, 3), padding='same', activation='relu')(x)
        x = MaxPool3D(pool_size=(1, 2, 2), strides=None)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=64, kernel_size=(3 , 3, 3), padding='same', activation='relu')(x)
        x = MaxPool3D(pool_size=(1, 2, 2), strides=None)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=128, kernel_size=(3 , 3, 3), padding='same', activation='relu')(x)
        x = MaxPool3D(pool_size=(2, 2, 2), strides=None)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=128, kernel_size=(3 , 3, 3), padding='same', activation='relu')(x)
        x = MaxPool3D(pool_size=(2, 2, 2), strides=None)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=256, kernel_size=(2 , 3, 3), padding='same', activation='relu')(x)
        x = MaxPool3D(pool_size=(2, 2, 2), strides=None)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=256, kernel_size=(1 , 3, 3), padding='same', activation='relu')(x)
        self.conv_shape = K.int_shape(x) #Shape of conv to be provided to decoder

        x = Flatten(name='latent_space')(x)
        x = Dense(200, activation='relu')(x)
        self.latent_space = Dense(self.latent_space_dim, activation='sigmoid')(x)
    
        self.encoder = Model(encoder_input, self.latent_space, name="encoder")

        self._model_input = encoder_input



if __name__ == "__main__":
    autoencoder = MCAE(
        input_shape=(8, 83, 82, 2),
        latent_space_dim = 200,
    )
    autoencoder.summary()


