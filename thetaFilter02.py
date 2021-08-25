import cv2
# opencv version 4.5.3.56
import numpy as np
# numpy version 1.21.1
import matplotlib.pyplot as plt

import os
import sys

class thetaFilter:
    def __init__(self, path):
        self.theta = 0
        self.delta_theta = 5
        self.image = cv2.imread(path)
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #self.image_gray = cv2.imread(path) #lee la imágen en gris

    # método set_theta --> recibe los parámetros 𝜃 y Δ𝜃 que definen la respuesta del filtro.
    def set_theta(self, theta, delta_theta):
        self.theta = theta
        self.delta_theta = delta_theta


    def filtering(self):
        image_gray_fft = np.fft.fft2(self.image_gray) #transformada de fourier
        image_fft_shift = np.fft.fftshift(image_gray_fft) #invierte cuadrantes

        # fft visualization
        image_gray_fft_mag = np.absolute(image_fft_shift) # magnitud
        image_fft_view = np.log(image_gray_fft_mag + 1) # funcion de logaritmo
        image_fft_view = image_fft_view / np.max(image_fft_view) # escalizar a 1
        #calculo el tamaño de la mascara
        mask = np.zeros_like(self.image_gray)

        num_rows, num_cols = np.shape(mask)  # numero de filas y columnas de la mascara
        enum_rows = np.linspace(0, num_rows - 1, num_rows)  # enumera filas
        enum_cols = np.linspace(0, num_cols - 1, num_cols)  # enumera columnas
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)  # crea malla
        half_size = num_rows / 2 - 1  # calcula punto medio

        # calcula todos los angulos sobre la malla
        id_angle = np.arctan2(col_iter - half_size, row_iter - half_size) * 180 / np.pi





        #crea la mascara
        theta_ = self.theta
        delta_ = self.delta_theta

        for i, j in enumerate(mask):
            for k, l in enumerate(mask):
                if (id_angle[i, k] <= (theta_) and id_angle[i, k] >= (theta_ - delta_)) or \
                        (id_angle[i, k] >= (theta_) and id_angle[i, k] <= (theta_) + delta_):
                        mask[i, k] = 1

        for i, j in enumerate(mask):
            for k, l in enumerate(mask):
                if theta_==0:
                    if (id_angle[i, k] >= (theta_ + 180) and id_angle[i, k] <= (theta_ + 180) + delta_) or \
                            (id_angle[i, k] <= ((theta_ + 180)) and id_angle[i, k] >= (theta_ + 180) - delta_):
                        mask[i, k] = 1
                else:
                    if (id_angle[i, k] >= (theta_ - 180) and id_angle[i, k] <= (theta_ - 180) + delta_) or \
                            (id_angle[i, k] <= ((theta_ - 180)) and id_angle[i, k] >= (theta_ - 180) - delta_):
                            mask[i, k] = 1

        print(mask)# imprimo la mascara

        fft_filter = image_fft_shift*mask #aqui va la mascara
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filter))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)
        title = "imagen3 "
        cv2.imshow("Image1", self.image_gray)
        cv2.imshow("Filter frequency response", 255 * mask)
        #cv2.imshow("Image2", image_fft_view)
        cv2.imshow(title, image_filtered)
        cv2.waitKey(0)



def banco4():
    img0 = thetaFilter(path_file)
    img0.set_theta(180, 5)
    img0.filtering()

    img45 = thetaFilter(path_file)
    img45.set_theta(45, 5)
    img45.filtering()

    img90 = thetaFilter(path_file)
    img90.set_theta(90, 5)
    img90.filtering()

    img135 = thetaFilter(path_file)
    img135.set_theta(135, 5)
    img135.filtering()





if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    #image_gray = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
    """
    prueba = thetaFilter(path_file)
    prueba.set_theta(45, 5)
    prueba.filtering()
    """
    banco4()



