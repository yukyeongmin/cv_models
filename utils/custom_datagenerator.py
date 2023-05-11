import os.path
import tensorflow as tf
import cv2
import numpy as np
import skimage


class SirtaDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_items, image_size, batch_size=32, shuffle=True):
        self.index = 0
        self.batch_size = batch_size
        self.image_size = image_size
        self.list_items = list_items        # actually this is dictionary {img path1, img path2, [additional info(elev, azi, error)]}
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_items) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_items[k] for k in indexes]

        # Generate data
        high_exposure, low_exposure, infos = self.__data_generation(list_IDs_temp)

        return high_exposure, low_exposure#, infos

    def __call__(self, *args, **kwargs):
        return self

    def next(self):
        self.index += 1
        return self.__getitem__(self.index-1)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_items))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples X : (n_samples, *dim, n_channels)"""

        if self.batch_size == 1:
            item = list_IDs_temp[0]
            high_exposure = cv2.imread(item["impath_high"])
            low_exposure = cv2.imread(item["impath_low"])
            high_exposures = self.preProcess(high_exposure)
            low_exposures = self.preProcess(low_exposure)
            infos = [item["elevation"], item["azimuth"]]

        else:
            # Initialization
            high_exposures = np.empty((self.batch_size, self.image_size[0], self.image_size[1], 3), dtype=float)
            low_exposures = np.empty((self.batch_size, self.image_size[0], self.image_size[1], 3), dtype=float)
            infos = np.empty((self.batch_size, 2), dtype=float)

            # Generate data
            for i, item in enumerate(list_IDs_temp):

                high_exposure = cv2.imread(item["impath_high"])
                low_exposure = cv2.imread(item["impath_low"])

                high_exposure = self.preProcess(high_exposure)
                high_exposures[i] = high_exposure
                low_exposure = self.preProcess(low_exposure)
                low_exposures[i] = low_exposure

                '''label =  elev , azimuth, error'''
                # Store class
                infos[i] = [item["elevation"], item["azimuth"]] #, item["error"]]

        return high_exposures, low_exposures, infos

    def preProcess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]), cv2.INTER_CUBIC)
        return img/255.


class DataGeneratorExpo(tf.keras.utils.Sequence):
    # mark version
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        fdim = [self.dim[1]*4 , self.dim[0]]

        X = np.empty((self.batch_size, *fdim, self.n_channels))
        y = np.empty((self.batch_size, 2), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            item = ID
            # Store sample
            imOrg = cv2.imread(item["impathorg"])/255
            imOrg= cv2.resize(imOrg, self.dim, interpolation=cv2.INTER_CUBIC)

            imExp = cv2.imread(item["impathexp"])/255
            imExp = cv2.resize(imExp, self.dim, interpolation=cv2.INTER_CUBIC)

            mag, theta = self.genGradIMUpdated(imOrg)

            fa = [imOrg, imExp, mag, theta]

            frame_concat = cv2.vconcat(fa)

            #X[i,] = frame_concat
            X[i,] = frame_concat

            '''label =  elev , azimuth'''


            # Store class
            y[i] = [item["elevation"] , item["azimuth"]]

        return X, y


        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def genGradIMUpdated(self, im):

        mag1, theta1 = self.genGradField(im[:, :, 0])
        mag2, theta2 = self.genGradField(im[:, :, 1])
        mag3, theta3 = self.genGradField(im[:, :, 2])

        #mag = np.dstack([mag1, mag2, mag3])
        mag = np.dstack([mag1, mag2, mag3])
        theta = np.dstack([theta1, theta2, theta3])

        return mag,theta

    def genGradField(self, im):

        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

        Ky = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])

        Ix = cv2.filter2D(im, -1, Kx)
        Iy = cv2.filter2D(im, -1, Ky)

        G = np.hypot(Ix, Iy)
        G = skimage.exposure.rescale_intensity(G, in_range='image', out_range=(0, 255)).astype(np.float64)

        theta = np.arctan2(Iy, Ix)
        theta = skimage.exposure.rescale_intensity(theta, in_range='image', out_range=(0, 255)).astype(np.float64)

        return G, theta

    def preProcess2(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # im = cv2.resize(im, (800,200), cv2.INTER_CUBIC)
        # gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5, sigmaY=1.5)
        return gray
