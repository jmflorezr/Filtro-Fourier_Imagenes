import cv2
# opencv version 4.5.3.56
import numpy as np
# numpy version 1.21.1

# Norbey  Marin Moreno
# Julian Mauricio Florez
#imagenes de prueba huellas


import os
import sys

class thetaFilter:
    def __init__(self, image):
        self.theta = 0
        self.delta_theta = 5
        self.image_gray = image

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
        half_size = (num_rows / 2)  # calcula punto medio

        # calcula todos los angulos sobre la malla
        id_angle = np.arctan2(col_iter - half_size, row_iter - half_size) *180/ np.pi

        #crea la mascara---------------------------------------------------
        theta_ = self.theta
        delta_ = self.delta_theta

        for i, j in enumerate(mask):
            for k, l in enumerate(mask):
                if theta_ != 180:
                    if (theta_ >= id_angle[i, k] >= (theta_ - delta_)) or \
                            (id_angle[i, k] >= (theta_) and id_angle[i, k] <= (theta_) + delta_):
                            mask[i, k] = 1
                else:
                    if (np.abs(id_angle[i, k]) <= theta_  and np.abs(id_angle[i, k]) >= ((theta_ ) - delta_)):
                        mask[i, k] = 1

        for i, j in enumerate(mask):
            for k, l in enumerate(mask):
                if theta_ != 0:
                    if (id_angle[i, k] >= (theta_ - 180) and id_angle[i, k] <= (theta_ - 180) + delta_) or \
                            (id_angle[i, k] <= ((theta_ - 180)) and id_angle[i, k] >= (theta_ - 180) - delta_):
                            mask[i, k] = 1

                else:
                    if (np.abs(id_angle[i, k]) <= theta_+180 and np.abs(id_angle[i, k]) >= ((theta_+180) - delta_)):
                        mask[i, k] = 1

        mask[150, 150] = 1
        #Aplica el filtro--------------------------------------------
        fft_filter = image_fft_shift*mask #aqui va la mascara
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filter))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)

        # Visualización ---------------------------------------------
        title1 = "Filt de " + str(theta_) + " grados"
        title2 = "Img filtrada a " + str(theta_) + " grados"
        cv2.imshow("Imagen de entrada", self.image_gray)
        cv2.imshow(title1, 255 * mask)
        cv2.imshow(title2, image_filtered)
        cv2.waitKey(0)
        return image_filtered


# metodo que crea un banco de 4 filtros [0, 45, 90, 135] grados, Δ𝜃 = 5
def banco4( image ):
    img0 = thetaFilter(image)
    img0.set_theta(180, 30)
    uno = img0.filtering()
    uno = uno-uno.min()
    uno = uno/uno.max()

    img45 = thetaFilter(image)
    img45.set_theta(45, 30)
    dos = img45.filtering()
    dos = dos - dos.min()
    dos = dos / dos.max()

    img90 = thetaFilter(image)
    img90.set_theta(90, 30)
    tres = img90.filtering()
    tres = tres - tres.min()
    tres = tres / tres.max()

    img135 = thetaFilter(image)
    img135.set_theta(135, 30)
    cuatro = img135.filtering()
    cuatro = cuatro - cuatro.min()
    cuatro = cuatro / cuatro.max()

    cv2.imshow("promedio de imagnes", ((uno+dos+tres+cuatro).astype(np.float)/4))
    cv2.imshow("imagen de entrada", image)
    cv2.waitKey(0)






if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)  # lee la imagen
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # pasa imagen a gris
    #llama al medoto que visualiza los 4 filtros
    banco4(image_gray)