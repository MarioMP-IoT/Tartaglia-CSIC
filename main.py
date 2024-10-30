"""Script para proyecto Tartaglia"""

import sys
import cv2
import copy
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import time
from scipy import signal
import numba as nb

from file_functions import *





class VideoPlayer(QtGui.QWidget):

    def __init__(self, app):
        super().__init__()
        self.setWindowTitle('Estudio imagen ultrasónica para Tartaglia')
        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.app=app
        # Create widgets
        self.video_button = QtGui.QPushButton('Open Video')
        self.video_button.setFixedWidth(200)
        self.video_label = QtGui.QLabel('')


        self.video_view = pg.PlotWidget()
        self.video_view.hideAxis('bottom')
        self.video_view.hideAxis('left')
        self.video_view.setAspectLocked(True)
        # Item para mostrar image data
        self.img_video = pg.ImageItem()
        self.img_video.setLevels((0, 255))

        self.video_view.addItem(self.img_video)
        self.video_view.invertY()




        self.filtered_image_view = pg.PlotWidget()
        self.filtered_image_view.hideAxis('bottom')
        self.filtered_image_view.hideAxis('left')
        self.filtered_image_view.invertY()
        self.filtered_image_view.setAspectLocked(True)
        # Item para mostrar image data
        self.img_filtered = pg.ImageItem()
        self.img_filtered.setLevels((0, 255))

        self.filtered_image_view.addItem(self.img_filtered)



        win = self.geometry()
        # Scroll Bar
        self.scroll = QtGui.QScrollBar(self)
        self.scroll.setGeometry(1, 1, win.width(), 25)
        self.scroll.setOrientation(QtCore.Qt.Horizontal)
        self.scroll.hide()
        # adding action to the scroll bar
        self.scroll.valueChanged.connect(lambda: self.video_review())

        # Create checkboxes
        self.checkbox1 = QtGui.QCheckBox('Frequency Compounded')
        self.checkbox1.setChecked(True) #Por defecto
        self.checkbox2 = QtGui.QCheckBox('Spatial Compounded')
        # self.checkbox3 = QtGui.QCheckBox('FIR')

        # Create button to open the configuration dialog
        self.config_button = QtGui.QPushButton("Configurar Filtros")
        #Create button to config spatial compounded
        self.spatial_config_button = QtGui.QPushButton("Configurar filt. espaciales")

        #Etiqueta para mostrar por pantalla las subbandas usadas
        self.subbands_label = QtGui.QLabel('Subbandas de frecuencia usadas: ')

        # Crear un ComboBox para seleccionar el modo de adquisición
        self.mode_combo = QtGui.QComboBox()
        self.mode_combo.addItem('Disparo Simple')
        self.mode_combo.addItem('Múltiple Disparo')
        # Establecer "Disparo Simple" como opción predeterminada
        self.mode_combo.setCurrentIndex(0)

        # Crear un botón para abrir la ventana de adquisiciones múltiples
        self.open_multi_acq_button = QtGui.QPushButton('Abrir Adquisiciones Múltiples')
        self.open_multi_acq_button.setEnabled(False)


        #crear boton de guardado de bscan procesado y de comparar bscan
        self.save_button = QtGui.QPushButton('Guardar')
        self.save_button.setEnabled(False)
        self.compare_button = QtGui.QPushButton('Comparar')
        self.compare_button.setEnabled(False)


        # Conectar la señal de cambio de selección en el ComboBox
        self.mode_combo.currentIndexChanged.connect(self.on_mode_selection_changed)

        # Create a horizontal layout for buttons
        button_layout = QtGui.QHBoxLayout()

        # Button to calculate standard deviation
        self.std_button = QtGui.QPushButton('Calcular Desviación Típica')
        self.std_button.clicked.connect(self.calculate_std)

        # Button to calculate contrast to noise ratio
        self.cnr_button = QtGui.QPushButton('Calcular Contraste al Ruido')
        self.cnr_button.clicked.connect(self.calculate_cnr)

        button_layout.addWidget(self.std_button)
        button_layout.addWidget(self.cnr_button)

        # Create a QWidget to hold the horizontal button layout
        self.button_widget = QtGui.QWidget()
        self.button_widget.setLayout(button_layout)

        #Buttons for filtered image

        button_layout_filt = QtGui.QHBoxLayout()

        # Button to calculate standard deviation
        self.std_button_filt = QtGui.QPushButton('Calcular Desviación Típica')
        self.std_button_filt.clicked.connect(self.calculate_std)

        # Button to calculate contrast to noise ratio
        self.cnr_button_filt = QtGui.QPushButton('Calcular Contraste al Ruido')
        self.cnr_button_filt.clicked.connect(self.calculate_cnr)

        button_layout_filt.addWidget(self.std_button_filt)
        button_layout_filt.addWidget(self.cnr_button_filt)
        # Create a QWidget to hold the horizontal button layout
        self.button_widget_filt = QtGui.QWidget()
        self.button_widget_filt.setLayout(button_layout_filt)

        #Labels to show std and cnr values
        labels_results_layout = QtGui.QHBoxLayout()

        self.std_label=QtGui.QLabel('STD: -')
        self.cnr_label = QtGui.QLabel('CNR: -')
        self.std_label_filt = QtGui.QLabel('STD: -')
        self.cnr_label_filt = QtGui.QLabel('CNR: -')

        labels_results_layout.addWidget(self.std_label)
        labels_results_layout.addWidget(self.cnr_label)
        labels_results_layout.addWidget(self.std_label_filt)
        labels_results_layout.addWidget(self.cnr_label_filt)

        # Create a QWidget to hold the horizontal button layout
        self.labels_result_widget = QtGui.QWidget()
        self.labels_result_widget.setLayout(labels_results_layout)



        # Estructura de la ventana

        layout = QtGui.QGridLayout()


        self.resize(1400, 1000)
        layout.addWidget(self.video_button,0,0,1,1)
        layout.addWidget(self.video_label,1,0,1,3)


        layout.addWidget(self.video_view,2,0,2,3)
        # self.video_view.setMinimumHeight(350)  # Añado tamaño minimo a la figura principal

        layout.addWidget(self.checkbox1,0,3,1,1)
        layout.addWidget(self.checkbox2,2,3,1,1)
        layout.addWidget(self.subbands_label,1,3,1,1)

        layout.addWidget(self.filtered_image_view, 3, 3, 1, 4)
        #layout.addWidget(self.scroll,6,0,1,7)
        #layout.addWidget(self.button_widget,4,0,1,1)
        layout.addWidget(self.button_widget_filt, 4, 0, 1, 7)
        layout.addWidget(self.labels_result_widget,5,0,1,7)
        layout.addWidget(self.mode_combo,0, 4, 1, 1)
        layout.addWidget(self.config_button, 0, 5, 1, 1)
        layout.addWidget(self.spatial_config_button, 2, 4, 1, 1)
        layout.addWidget(self.open_multi_acq_button, 0, 1, 1, 1)


        layout.addWidget(self.save_button, 0, 2, 1, 1)
        layout.addWidget(self.compare_button, 0, 6, 1, 1)


        layout.setColumnStretch(0, 1)  # Columna 0 se agranda igual que la columna 1
        layout.setColumnStretch(1, 1)  # Columna 1 se agranda igual que la columna 0
        # layout.setColumnStretch(2, 0)  # Columna 2 no se agranda
        # layout.setColumnStretch(3, 0)  # Columna 3 no se agranda


        layout.setRowStretch(0, 0)  # Fila 0 no se agranda
        # layout.setRowStretch(1, 1)  # Fila 1 se agranda igual que la fila 2
        layout.setRowStretch(2, 1)  # Fila 2 se agranda igual que la fila 1

        # Set layout
        self.setLayout(layout)

        # Connect buttons to functions
        self.video_button.clicked.connect(self.open_video_file)
        self.config_button.clicked.connect(self.open_config_dialog)
        self.spatial_config_button.clicked.connect(self.open_spatial_config_dialog)
        self.open_multi_acq_button.clicked.connect(self.open_multi_acq_dialog)

        self.compare_button.clicked.connect(self.compare_function)
        self.save_button.clicked.connect(self.save_function)



        # # Connect checkbox signals to slots
        self.checkbox1.stateChanged.connect(self.toggle_checkboxes)
        self.checkbox2.stateChanged.connect(self.toggle_checkboxes)
        #self.checkbox3.stateChanged.connect(self.update_images)

        # Create an instance of FrequencyCompoundingUI
        self.config_dialog = FrequencyCompoundingUI()
        # Create an instance of SpatialCompoundingUI
        self.spatial_config_dialog = SpatialCompoundingUI()




        # Initialize variables
        self.video_file = ''
        self.video_capture = None
        self.video_frame = None

        self.roi_mask = None
        self.image=None

        self.background_roi = None
        self.contrast_roi = None
        self.std_roi = None

        self.prev_scroll_pos=0
        self.frame_pos=0
        self.comparing = 0 #bandera para diferenciar si estoy comparando imagenes o no
        #valores filtro por defecto
        self.n_subbands = 1
        self.central_frequency = 3.5e6
        self.total_bandwidth = 3e6
        self.overlap_frequency = 0

        #valores spatialcompounding por defecto
        self.n_steps=5
        self.n_elements=64
        self.aperture=48





    def clearWindows(self):

        # Initialize variables
        self.video_file = ''
        self.video_capture = None
        self.video_frame = None

        self.prev_scroll_pos = 0
        self.frame_pos = 0
        # self.background_roi = None
        # self.contrast_roi = None
        # self.std_roi=None



        # self.p0 = None
    def open_video_file(self):
        if self.video_capture is not None:
            self.clearWindows()
            self.save_button.setEnabled(False)
            self.comparing = 0

        "funcion para cargar los .bin"



        self.video_file, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open Video', '', 'Video Files (*.BIN) ;; PNG files (*.png)')
        self.extension=self.video_file.split('.')[-1]
        if self.video_file and (self.extension=='BIN' or self.extension == 'bin'):
            # Leo el archivo y genero un objeto 'BinFile' que contiene los datos crudos tal cual se leen del archivo
            bfile = BinFile(self.video_file)
            # A partir del objeto 'BinFile' genero un objeto 'Dataset' que ya tiene los datos en formato 'cubo de datos'
            # y contiene las matrices de coordenadas de los píxeles para poder pintar
            self.dset = Dataset(bfile, resample=0, med_kernel=0, avg_kernel=0)


            #for i in range(self.dset.nimg):

            envolvente = signal.hilbert(self.dset.bscan[:,:,0], axis=0)
            # coeff = [-0.0018790103, -0.0003538003, -0.0018397546, -0.0006974823, -0.0027199950, -0.0012019279,
            #          -0.0038480745, -0.0019126439, -0.0052654910, -0.0028847239, -0.0070205066, -0.0041862177,
            #          -0.0091777238, -0.0059058206, -0.0118248552, -0.0081646622, -0.0150887042, -0.0111409845,
            #          -0.0191715703, -0.0151178496, -0.0244217110, -0.0205841472, -0.0314885954, -0.0284813086,
            #          -0.0417299535, -0.0409085813, -0.0585098994, -0.0637067963, -0.0929410511, -0.1214409414,
            #          -0.2155478257, -0.6319739537, 0.6319739537, 0.2155478257, 0.1214409414, 0.0929410511, 0.0637067963,
            #          0.0585098994, 0.0409085813, 0.0417299535, 0.0284813086, 0.0314885954, 0.0205841472, 0.0244217110,
            #          0.0151178496, 0.0191715703, 0.0111409845, 0.0150887042, 0.0081646622, 0.0118248552, 0.0059058206,
            #          0.0091777238, 0.0041862177, 0.0070205066, 0.0028847239, 0.0052654910, 0.0019126439, 0.0038480745,
            #          0.0012019279, 0.0027199950, 0.0006974823, 0.0018397546, 0.0003538003, 0.0018790103]
            # envolvente = self.hilbert_transform(self.dset.bscan[:,:,0], coeff)
            envolvente = 20 * np.log10(np.abs(envolvente) / 3276.7)  # TAMAÑO INT16 pasamos a dB
            envolvente[envolvente< -self.dset.dB_range] = -self.dset.dB_range
            self.dset.bscan[:, :, 0]=envolvente


            self.dset.ScanConvert(raw=False) #como bscan y raw son iguales trabajo con uno para filtrar y el otro pinto para comparar


            self.video_label.setText('Video: ' + self.video_file)
            # self.frames = []
            self.scroll.setRange(0, self.dset.nimg - 1)  # Añado rango del scroll en funcion del numero de imagenes
            self.scroll.setSliderPosition(0)


            self.update_images()

            self.first = True  # bandera para poder cargar imagen to track la primera vez
            self.video_openned = True
            #self.scroll.show()
            del bfile
        elif self.video_file and (self.extension == 'png' or self.extension == 'PNG'):
            file_path = self.video_file
            # Cargar la imagen en color
            imagen_color = cv2.imread(file_path)

            # Convertir la imagen a escala de grises
            imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
            pixmap = QtGui.QPixmap(file_path)
            self.image = np.array(imagen_gris)#np.array(pixmap.toImage())
            self.img_video.setImage(self.image)



    def open_config_dialog(self):

        # Load the configuration values before showing the dialog
        self.config_dialog.subbands_spinbox.setValue(self.n_subbands)
        self.config_dialog.central_frequency_spinbox.setValue(self.central_frequency/10**6)
        self.config_dialog.bandwidth_spinbox.setValue(self.total_bandwidth/10**6)
        self.config_dialog.overlap_spinbox.setValue(self.overlap_frequency/10**6)

        if self.config_dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Apply the settings from the dialog
            self.n_subbands = int(self.config_dialog.subbands_spinbox.value())
            self.central_frequency = self.config_dialog.central_frequency_spinbox.value() * 10**6
            self.total_bandwidth = self.config_dialog.bandwidth_spinbox.value()* 10**6
            self.overlap_frequency=self.config_dialog.overlap_spinbox.value()* 10**6
            # Apply the frequency compounded imaging with the settings
            # Replace this with your actual frequency compounded imaging function
            #compounded_image_data = self.frequency_compounded_imaging(subbands, central_frequency, bandwidth)
            # Update the images in the PlotWidgets
            #self.raw_image.setImage(self.get_raw_image_data())  # Replace this with your raw image data
            # self.img_filtered.setImage(compounded_image_data)
            if self.video_file != '':
                self.update_images(only_filtered=False)# TODO: only_fltered es la bandera que habilita o no mostral la imagen sin filtro

    def open_spatial_config_dialog(self):

        # Load the configuration values before showing the dialog
        self.spatial_config_dialog.n_step_spinBox.setValue(self.n_steps)
        self.spatial_config_dialog.aperture_spinBox.setValue(self.aperture)
        self.spatial_config_dialog.n_elements_spinBox.setValue(self.n_elements)


        if self.spatial_config_dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Apply the settings from the dialog
            self.n_steps = int(self.spatial_config_dialog.n_step_spinBox.value())
            self.aperture = self.spatial_config_dialog.aperture_spinBox.value()
            self.n_elements = self.spatial_config_dialog.n_elements_spinBox.value()
            self.last_el=self.spatial_config_dialog.last_spinBox.value()
            self.first_el = self.spatial_config_dialog.first_spinBox.value()
            self.deltaMax =(self.spatial_config_dialog.last_spinBox.value() - self.spatial_config_dialog.fist_spinBox.value())* self.spatial_config_dialog.pitch.value() #valor en mm
            #self.c_



            if self.video_file != '':
                self.update_images(only_filtered=True)


    def on_mode_selection_changed(self):
        # Habilitar o deshabilitar el botón de adquisiciones múltiples según la selección
        if self.mode_combo.currentIndex() == 1:  # 1 corresponde a "Múltiple Disparo"
            self.open_multi_acq_button.setEnabled(True)
            self.config_button.setEnabled(False)
            self.video_button.setEnabled(False)
            self.clearWindows()
        else:
            self.open_multi_acq_button.setEnabled(False)
            self.config_button.setEnabled(True)
            self.video_button.setEnabled(True)
            self.clearWindows()

    def video_review(self):
        # getting current value of scroll bar
        frame_pos = self.scroll.value()
        self.frame_pos = frame_pos


        #llamo a la funcion de pintado para actualizar las imagenes
        self.update_images()

    def toggle_checkboxes(self, state):
        sender = self.sender()  # Obtiene el objeto que emitió la señal

        if state == 2:
            if sender == self.checkbox1:
                self.checkbox2.setChecked(False)
                if self.mode_combo.currentIndex() == 1:
                    self.open_multi_acq_button.setEnabled(False)
                else:
                    self.open_multi_acq_button.setEnabled(True)
                #self.update_images()
            else:
                self.checkbox1.setChecked(False)
                self.open_multi_acq_button.setEnabled(True)
                self.config_button.setEnabled(False)
                self.mode_combo.setEnabled(False)
                #self.update_images()

    def update_images(self, only_filtered=False):


        if self.checkbox1.isChecked():

            if not only_filtered:
                self.image = self.dset.frames[:, :, self.frame_pos]

                self.img_video.setImage(self.image)
            if not self.video_file == '':
                #Frequency compounded
                self.compound_image()
                self.dset.ScanConvert(filtered_frame=self.bscan_filtered[:, :, self.frame_pos])
                self.image_filtered = self.dset.filt_frame
                self.img_filtered.setImage(self.image_filtered, autoLevels=False)

        elif self.checkbox2.isChecked():

            if not only_filtered:
                self.image = self.dset.frames[:, :, self.frame_pos]

                self.img_video.setImage(self.image)
            if not self.video_file == '':
                # TODO spatial compounded
                self.compound_image()
                self.dset.ScanConvert(filtered_frame=self.bscan_filtered[:, :, self.frame_pos])
                self.image_filtered = self.dset.filt_frame
                self.img_filtered.setImage(self.image_filtered, autoLevels=False)



    def compound_image(self):
        # Copy the original image
        processed_image = self.dset.raw_bscan[:,:,self.frame_pos].copy()
        empty=False
        self.bscan_filtered=np.zeros_like(self.dset.bscan)

        # Apply filters based on checkbox states
        if self.checkbox1.isChecked(): #frequency compounded
            print('AQUI VA FILTRO')
            #Calculamos las subbandas



            fs = 40e6 # Frecuencia de muestreo en Hz
            self.subbands=calculate_subbands(self.total_bandwidth, self.n_subbands, self.overlap_frequency, self.central_frequency)
            self.subbands_string = ",".join(f"({start / 1e6:.2f}-{end / 1e6:.2f}) MHz" for start, end in self.subbands)
            print(self.subbands_string)
            self.subbands_label.setText('Subbandas de frecuencia usadas: '+ self.subbands_string)

            self.filt_sub_bscans=np.zeros((self.dset.nr,self.dset.nlin,self.n_subbands))

            processed_image= self.frequency_compounded_imaging(processed_image, fs)

            #abro ventana emergente con imagenes parciales

            self.simple_win=SimpleAcquisitionWindow(self)



        elif self.checkbox2.isChecked():#spatial compounded
            print('AQUI VA spatial compounded')
            #aqui se deberia de hacer llamada al metodo spatial compounding
            #processed_image = self.spatial_compounded_imaging(processed_image)


        # elif self.checkbox3.isChecked(): #FIR
        #     print('AQUI VA FILTRO')
        #     processed_image=FIR_linear_filter(processed_image)
        else:
            empty=True

        #Guardo frame procesado

        if empty:
            envolvente = signal.hilbert(processed_image, axis=0)
            envolvente = 20 * np.log10(np.abs(envolvente) / 3276.7)  # TAMAÑO INT16 pasamos a dB
            envolvente[envolvente < -self.dset.dB_range] = -self.dset.dB_range
            # change the processed image
            self.bscan_filtered[:,:,self.frame_pos] = envolvente
        else:
            # change the processed image
            self.bscan_filtered[:,:,self.frame_pos]=processed_image

    def hilbert_transform(self, image, hilbert_coeffs):
        # Aplica los coeficientes precalculados a la señal mediante convolución
        hilbert_signal =  np.apply_along_axis(lambda x: np.convolve(x, hilbert_coeffs, mode='same'), 0, image)
        return hilbert_signal
    #@jit(nopython=True, parallel=False, cache=True)
    def frequency_compounded_imaging(self, b_scan, fs):
        num_subbands = self.n_subbands

        # compounded_image, self.filt_sub_bscans = compounding_image(self.subbands, b_scan, fs, self.dset.dB_range)
        num_samples, num_lines = b_scan.shape

        # Inicializar la imagen compuesta
        compounded_image = np.zeros((num_samples, num_lines))
        i=0 #indice para saber que subbanta se filtra
        # Aplicar Frequency Compounded Imaging
        for start, end in self.subbands:
            # Diseñar el filtro paso banda

            b = signal.firwin(128, [start, end], pass_zero='bandpass', fs=fs, scale=True)
            a = 1
            # Filtrar la matriz cruda B-scan con el filtro paso banda
            filtered_b_scan = signal.filtfilt(b, a, b_scan, axis=0)
            # envolvente= np.abs(filtered_b_scan)
            envolvente = signal.hilbert(filtered_b_scan, axis=0)
            #coeff=[-0.0018790103,-0.0003538003,-0.0018397546,-0.0006974823,-0.0027199950,-0.0012019279,-0.0038480745,-0.0019126439,-0.0052654910,-0.0028847239,-0.0070205066,-0.0041862177,-0.0091777238,-0.0059058206,-0.0118248552,-0.0081646622,-0.0150887042,-0.0111409845,-0.0191715703,-0.0151178496,-0.0244217110,-0.0205841472,-0.0314885954,-0.0284813086,-0.0417299535,-0.0409085813,-0.0585098994,-0.0637067963,-0.0929410511,-0.1214409414,-0.2155478257,-0.6319739537 ,0.6319739537 ,0.2155478257 ,0.1214409414 ,0.0929410511 ,0.0637067963 ,0.0585098994 ,0.0409085813 ,0.0417299535 ,0.0284813086 ,0.0314885954 ,0.0205841472 ,0.0244217110 ,0.0151178496 ,0.0191715703 ,0.0111409845 ,0.0150887042 ,0.0081646622 ,0.0118248552 ,0.0059058206 ,0.0091777238 ,0.0041862177 ,0.0070205066 ,0.0028847239 ,0.0052654910 ,0.0019126439 ,0.0038480745 ,0.0012019279 ,0.0027199950 ,0.0006974823 ,0.0018397546 ,0.0003538003 ,0.0018790103]
            #envolvente = self.hilbert_transform(filtered_b_scan, coeff)
            envolvente = 20 * np.log10(np.abs(envolvente)/3276.7)  # TAMAÑO INT16 pasamos a dB
            envolvente[envolvente < -self.dset.dB_range] = -self.dset.dB_range

            # Acumular la sub-banda filtrada en la imagen compuesta
            compounded_image += envolvente
            #-----------
            # intermedia= 20 * np.log10(envolvente/3276.7)  # TAMAÑO INT16 pasamos a dB
            # intermedia[intermedia < -self.dset.dB_range] = -self.dset.dB_range
            #---------------
            self.filt_sub_bscans[:,:,i]=envolvente
            i += 1
        # Normalizar la imagen compuesta
        compounded_image /= num_subbands

        #aplico filtro paso bajo equivalente a t.hilbert
        # b,a=signal.butter(6, 5*10**6, fs=fs, btype='low', analog=False)
        # compounded_image= signal.lfilter(b, a, compounded_image)
        # compounded_image = 20 * np.log10(np.abs(compounded_image) / 3276.7)  # TAMAÑO INT16 pasamos a dB
        # compounded_image[compounded_image < -self.dset.dB_range] = -self.dset.dB_range

        # # compounded_image=(compounded_image)/np.min(compounded_image)*(self.dset.dB_range) -self.dset.dB_range#convierto a valores de -48 a 0

        return compounded_image


    def open_spatial_compounded_imaging(self):
        self.spatial_dialog=SpatialWindow(self)
        self.spatial_dialog.exec_()

    def calculate_std(self):
        # habilito funcion de guardado y de comparar
        self.save_button.setEnabled(True)
        self.compare_button.setEnabled(True)
        multiple_shot = self.mode_combo.currentIndex()
        if multiple_shot == 0and not self.comparing and not self.checkbox2.isChecked():
            # self.bscan_data_filt = self.bscan_filtered[:,:,self.frame_pos]
            #
            # self.bscan_data = self.dset.bscan[:,:,self.frame_pos]
            self.bscan_data_filt = self.simple_win.compounded_bscan
            self.bscan_data = self.image

            # Create the ROI dialog and show it
            self.std_roi_dialog = STD_ROIDialog(multiple_shot, self.bscan_data, self.bscan_data_filt, self)
            self.std_roi_dialog.exec_()
            self.std_roi = self.std_roi_dialog.std_roi
            # # calculate the standard deviation of the image data
            #
            # std_value_filt = np.std(self.bscan_filtered[:,:,self.frame_pos])
            #
            # std_value = np.std(self.dset.bscan[:,:,self.frame_pos])
            #
            # self.std_label.setText("STD: {:.2f}".format(std_value))
            # self.std_label_filt.setText("STD: {:.2f}".format(std_value_filt))
        elif multiple_shot == 0 and self.comparing and not self.checkbox2.isChecked():
            # Create the ROI dialog and show it
            # self.bscan_data_filt = self.bscan_filtered[:, :, self.frame_pos]

            # self.bscan_data = elf.dset.bscan[:,:,self.frame_pos]
            self.bscan_data_filt = self.simple_win.compounded_bscan
            self.bscan_data = self.image
            self.std_roi_dialog = STD_ROIDialog(multiple_shot, self.bscan_data, self.bscan_data_filt, self)
            self.std_roi_dialog.exec_()
            self.std_roi = self.std_roi_dialog.std_roi
            # # calculate the standard deviation of the image data
            #
            # std_value_filt = np.std(self.bscan_filtered[:, :, self.frame_pos])
            #
            # std_value = np.std(self.bscan_data)
            #
            # self.std_label.setText("STD: {:.2f}".format(std_value))
            # self.std_label_filt.setText("STD: {:.2f}".format(std_value_filt))
        elif multiple_shot == 1 and self.comparing and not self.checkbox2.isChecked():
            self.bscan_data_filt = self.multi_dialog.compounded_bscan

            # Create the ROI dialog and show it
            self.std_roi_dialog = STD_ROIDialog(multiple_shot, self.bscan_data, self.bscan_data_filt, self)
            self.std_roi_dialog.exec_()
            self.std_roi = self.std_roi_dialog.std_roi
            # # calculate the standard deviation of the image data
            #
            # std_value_filt = np.std(self.multi_dialog.compounded_bscan)
            #
            # std_value = np.std(self.bscan_data)
            #
            # self.std_label.setText("STD: {:.2f}".format(std_value))
            # self.std_label_filt.setText("STD: {:.2f}".format(std_value_filt))

        else:
            if self.checkbox1.isChecked():
                self.bscan_data_filt = self.multi_dialog.compounded_bscan
            elif self.checkbox2.isChecked():
                self.bscan_data_filt = self.spatial_dialog.compounded_image
            self.bscan_data=None
            # Create the ROI dialog and show it
            self.std_roi_dialog = STD_ROIDialog(multiple_shot, self.bscan_data, self.bscan_data_filt, self)
            self.std_roi_dialog.exec_()
            self.std_roi = self.std_roi_dialog.std_roi
            # # calculate the standard deviation of the image data
            #
            # std_value_filt = np.std(self.multi_dialog.compounded_bscan)
            #
            # self.std_label_filt.setText("STD: {:.2f}".format(std_value_filt))
        #print(f"Desviación Típica: {std_value}")

    def calculate_cnr(self):
        # habilito funcion de guardado y de comparar
        self.save_button.setEnabled(True)
        self.compare_button.setEnabled(True)
        multiple_shot=self.mode_combo.currentIndex()

        if not multiple_shot and not self.comparing and not self.checkbox2.isChecked():
            #self.bscan_data_filt = self.bscan_filtered[:,:,self.frame_pos]
            #
            # self.bscan_data = self.dset.bscan[:,:,self.frame_pos]
            if not (self.extension == 'png' or self.extension == 'PNG'):
                self.bscan_data_filt=self.simple_win.compounded_bscan
            self.bscan_data = self.image

            # Create the ROI dialog and show it
            self.roi_dialog = CNR_ROIDialog(multiple_shot,self.bscan_data,self.bscan_data_filt, self)
            self.roi_dialog.exec_()
            self.background_roi = self.roi_dialog.background_roi
            self.contrast_roi = self.roi_dialog.interest_roi
        elif not multiple_shot and self.comparing and not self.checkbox2.isChecked():

            #self.bscan_data_filt = self.bscan_filtered[:, :, self.frame_pos]

            #self.bscan_data = self.bscan_data
            self.bscan_data_filt = self.simple_win.compounded_bscan
            self.bscan_data = self.image

            # Create the ROI dialog and show it
            self.roi_dialog = CNR_ROIDialog(multiple_shot, self.bscan_data, self.bscan_data_filt, self)
            self.roi_dialog.exec_()
            self.background_roi = self.roi_dialog.background_roi
            self.contrast_roi = self.roi_dialog.interest_roi
        elif multiple_shot and self.comparing and not self.checkbox2.isChecked():

            self.bscan_data_filt = self.multi_dialog.compounded_bscan

            # Create the ROI dialog and show it
            self.roi_dialog = CNR_ROIDialog(multiple_shot, self.bscan_data, self.bscan_data_filt, self)
            self.roi_dialog.exec_()
            self.background_roi = self.roi_dialog.background_roi
            self.contrast_roi = self.roi_dialog.interest_roi
        else:
            if self.checkbox1.isChecked():
                self.bscan_data_filt=self.multi_dialog.compounded_bscan
            elif self.checkbox2.isChecked():
                self.bscan_data_filt = self.spatial_dialog.compounded_image

            self.bscan_data=None
            # Create the ROI dialog and show it
            self.roi_dialog = CNR_ROIDialog(multiple_shot, self.bscan_data, self.bscan_data_filt, self)
            self.roi_dialog.exec_()
            self.background_roi = self.roi_dialog.background_roi
            self.contrast_roi = self.roi_dialog.interest_roi

    def open_multi_acq_dialog(self):

        if self.checkbox1.isChecked():
            # Create the ROI dialog and show it
            self.multi_dialog = MultiAcquisitionWindow(self)
            self.multi_dialog.exec_()
        elif self.checkbox2.isChecked():

            self.spatial_dialog = SpatialWindow(self)
            self.spatial_dialog.exec_()

    def save_function(self):
        path_save='.\\Results\\'

        np.save(path_save+ self.subbands_string + '.npy', np.float16(self.bscan_data_filt))

    def compare_function(self):
        self.comparing=1
        bscan_file, _ = QtGui.QFileDialog.getOpenFileName(self, 'Select file to compare', '', 'Bscan numpy files (*.npy)')
        if bscan_file:
            self.video_label.setText('Imagen: ' + bscan_file)
            self.bscan_data =np.load(bscan_file, allow_pickle=True).astype(float)
            if self.mode_combo.currentIndex() == 0:
                self.frame = np.ones((self.dset.frame_ny, self.dset.frame_nx), dtype='h') * (-self.dset.dB_range.astype('h'))  # frame compuesto

                ScanConvert_BilinealFrame(self.dset.frame_ny, self.dset.frame_nx, self.dset.frames_val, self.frame,
                                          self.bscan_data, self.dset.ix_test, self.dset.w_test)

                self.img_video.setImage(self.frame)
            else:
                self.frame = np.ones((self.multi_dialog.dset.frame_ny, self.multi_dialog.dset.frame_nx), dtype='h') * (
                    -self.multi_dialog.dset.dB_range.astype('h'))  # frame compuesto

                ScanConvert_BilinealFrame(self.multi_dialog.dset.frame_ny, self.multi_dialog.dset.frame_nx, self.multi_dialog.dset.frames_val, self.frame,
                                          self.bscan_data, self.multi_dialog.dset.ix_test, self.multi_dialog.dset.w_test)

                self.img_video.setImage(self.frame)


# from numba import jit, prange
#
# @jit(nopython=True, parallel=True)
# def hilbert(x):
#     N = x.shape[0]
#     X = np.fft.fft(x, N)
#     h = np.zeros(N)
#     h[0] = 1
#     h[1:N//2] = 2
#     if N % 2 == 0:
#         h[N//2] = 1
#     x = np.fft.ifft(X*h)
#     return x
#
# @jit(nopython=True, parallel=True)
# def compounding_image(subbands, b_scan, fs, dB_range):
#     num_samples, num_lines = b_scan.shape
#     num_subbands = len(subbands)
#
#     compounded_image = np.zeros((num_samples, num_lines))
#     filt_sub_bscans= np.zeros((num_samples, num_lines, num_subbands))
#
#     for i in prange(num_subbands):
#         start, end = subbands[i]
#
#         # Diseñar el filtro paso banda
#         freq = np.linspace(0, fs, 32)
#         b = np.zeros(32)
#         b[(freq >= start) & (freq <= end)] = 1
#         a = 1
#
#         # Filtrar la matriz cruda B-scan con el filtro paso banda
#         filtered_b_scan = np.zeros_like(b_scan)
#         for j in prange(num_lines):
#             filtered_b_scan[:, j] = np.convolve(b_scan[:, j], b)[:num_samples]
#
#         envolvente = np.zeros_like(filtered_b_scan)
#         for j in prange(num_lines):
#             envolvente[:, j] = np.abs(hilbert(filtered_b_scan[:, j]))
#         envolvente = 20 * np.log10(envolvente / 3276.7 + 1e-10)# TAMAÑO INT16 pasamos a dB
#         envolvente[envolvente < -dB_range] = -dB_range
#
#         # Acumular la sub-banda filtrada en la imagen compuesta
#         compounded_image += envolvente
#
#         filt_sub_bscans[:, :, i] = envolvente
#
#     return (compounded_image / num_subbands), filt_sub_bscans



def calculate_subbands(total_bandwidth, num_subbands, overlap_frequency, central_frequency):

    subband_step=total_bandwidth/ num_subbands


    subbands = []
    half_bandwidth = total_bandwidth / 2
    first_subband_start  = central_frequency - half_bandwidth
    for i in range(int(num_subbands)):
        start_freq=first_subband_start + subband_step*i
        if not i==0:
            start_freq=start_freq-overlap_frequency/2

        end_freq = first_subband_start+subband_step*(i+1)
        if not i==int(num_subbands)-1: #mientras no sea el ultimo caso
            end_freq=end_freq+overlap_frequency/2

        subbands.append((start_freq, end_freq))

    return subbands
class FrequencyCompoundingUI(QtWidgets.QDialog):

    def __init__(self):
        super().__init__()



        # Create form layout for settings
        form_layout = QtWidgets.QFormLayout(self)

        # SpinBoxes for configuring settings
        self.subbands_spinbox = QtGui.QDoubleSpinBox()
        self.subbands_spinbox.setMinimum(1)
        self.subbands_spinbox.setMaximum(9)
        self.subbands_spinbox.setValue(1)
        form_layout.addRow(QtGui.QLabel("Cantidad de Subbandas:"), self.subbands_spinbox)

        self.central_frequency_spinbox = QtGui.QDoubleSpinBox()
        self.central_frequency_spinbox.setMinimum(1)
        self.central_frequency_spinbox.setMaximum(10)
        self.central_frequency_spinbox.setValue(3.5)
        self.central_frequency_spinbox.setSingleStep(0.25)
        form_layout.addRow(QtGui.QLabel("Frecuencia Central (MHz):"), self.central_frequency_spinbox)

        self.bandwidth_spinbox = QtGui.QDoubleSpinBox()
        self.bandwidth_spinbox.setMinimum(0.1)
        self.bandwidth_spinbox.setMaximum(10)
        self.bandwidth_spinbox.setValue(3)
        self.bandwidth_spinbox.setSingleStep(0.25)
        form_layout.addRow(QtGui.QLabel("Ancho de Banda (MHz):"), self.bandwidth_spinbox)
        self.overlap_spinbox = QtGui.QDoubleSpinBox()
        self.overlap_spinbox.setMinimum(0)
        self.overlap_spinbox.setMaximum(2)
        self.overlap_spinbox.setValue(0)
        self.overlap_spinbox.setSingleStep(0.1)
        form_layout.addRow(QtGui.QLabel("Solapamiento en frecuencia (MHz):"), self.overlap_spinbox)

        # Button box for OK and Cancel buttons
        button_box = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        form_layout.addRow(button_box)

    def accept(self):

        super().accept()

    def reject(self):
        super().reject()

class SpatialCompoundingUI(QtWidgets.QDialog):

    def __init__(self):
        super().__init__()



        # Create form layout for settings
        form_layout = QtWidgets.QFormLayout(self)

        # SpinBoxes for configuring settings
        self.n_elements_spinBox = QtGui.QDoubleSpinBox()
        self.n_elements_spinBox.setMinimum(1)
        self.n_elements_spinBox.setMaximum(124)
        self.n_elements_spinBox.setValue(64)
        self.n_elements_spinBox.setSingleStep(1)
        form_layout.addRow(QtGui.QLabel("Elementos:"), self.n_elements_spinBox)

        self.aperture_spinBox = QtGui.QDoubleSpinBox()
        self.aperture_spinBox.setMinimum(1)
        self.aperture_spinBox.setMaximum(64)
        self.aperture_spinBox.setValue(48)
        self.aperture_spinBox.setSingleStep(1)
        form_layout.addRow(QtGui.QLabel("Apertura subimagen:"), self.aperture_spinBox)

        self.n_step_spinBox = QtGui.QDoubleSpinBox()
        self.n_step_spinBox.setMinimum(1)
        self.n_step_spinBox.setMaximum(20)
        self.n_step_spinBox.setValue(5)
        self.n_step_spinBox.setSingleStep(1)
        form_layout.addRow(QtGui.QLabel("Numero de subimagenes:"), self.n_step_spinBox)
        self.first_spinbox = QtGui.QDoubleSpinBox()
        self.first_spinbox.setMinimum(1)
        self.first_spinbox.setMaximum(16)
        self.first_spinbox.setValue(1)
        self.first_spinbox.setSingleStep(1)
        form_layout.addRow(QtGui.QLabel("Primer elemento:"), self.first_spinbox)
        self.last_spinbox = QtGui.QDoubleSpinBox()
        self.last_spinbox.setMinimum(1)
        self.last_spinbox.setMaximum(16)
        self.last_spinbox.setValue(16)
        self.last_spinbox.setSingleStep(1)
        form_layout.addRow(QtGui.QLabel("Ultimo elemento:"), self.last_spinbox)
        self.pitch = QtGui.QDoubleSpinBox()
        self.pitch.setDecimals(3)
        self.pitch.setMinimum(0.20)
        self.pitch.setMaximum(0.90)
        self.pitch.setValue(0.34)
        self.pitch.setSingleStep(0.01)
        form_layout.addRow(QtGui.QLabel("Pitch (mm):"), self.pitch)

        # Button box for OK and Cancel buttons
        button_box = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        form_layout.addRow(button_box)

    def accept(self):



        super().accept()

    def reject(self):
        super().reject()

class CNR_ROIDialog(QtWidgets.QDialog):
    def __init__(self,multiple_shot, image_data=None, image_data_filt=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Calculo de CNR')
        self.image_data = image_data
        self.image_data_filt=image_data_filt
        self.parent=parent
        self.multiple_shot=multiple_shot
        layout = QtWidgets.QVBoxLayout()
        self.video_view = pg.PlotWidget()
        self.video_view.hideAxis('bottom')
        self.video_view.hideAxis('left')
        self.video_view.setAspectLocked(False)
        # Item para mostrar image data



        self.image_view = pg.ImageItem()
        self.video_view.addItem(self.image_view)
        self.video_view.invertY()
        #self.image_view.setAspectLocked(False)
        if multiple_shot or  self.parent.checkbox2.isChecked():
            self.image_view.setImage(self.image_data_filt)
        else:
            self.image_view.setImage(self.image_data)

        self.roi_button = QtGui.QPushButton('ROI background')
        self.roi_button.clicked.connect(self.add_background_roi)

        self.roi_button2 = QtGui.QPushButton('ROI contrast')
        self.roi_button2.clicked.connect(self.add_interest_roi)

        self.calculate_button = QtGui.QPushButton('Calculate')
        self.calculate_button.clicked.connect(self.calculate)

        layout.addWidget(self.video_view)
        layout.addWidget(self.roi_button)
        layout.addWidget(self.roi_button2)
        layout.addWidget(self.calculate_button)

        self.setLayout(layout)

        self.background_roi = self.parent.background_roi #{'pos': Point(32.165493, 5262.965405), 'size': Point(6.582061, 814.354380), 'angle': 0.0}
        if not self.background_roi == None:
            self.roi_button.setStyleSheet("background-color: green")
            self.video_view.addItem(self.background_roi)

        self.interest_roi = self.parent.contrast_roi #{'pos': Point(32.165493, 5262.965405), 'size': Point(6.582061, 814.354380), 'angle': 0.0}
        if not self.interest_roi == None:
            self.roi_button2.setStyleSheet("background-color: green")
            self.video_view.addItem(self.interest_roi)

    def add_background_roi(self):
        # Implement the logic to add ROI to the background
        # For demonstration purposes, let's create a rectangle at the top-left corner
        self.background_roi = pg.ROI(pg.Point(58.794870, 3704.024460), pg.Point(4.427082, 541.539715), pen='r')#pg.ROI(pg.Point(32.165493, 5262.965405),pg.Point(6.582061, 814.354380), pen='r'){'pos': Point(58.794870, 3704.024460), 'size': Point(4.427082, 541.539715), 'angle': 0.0}
        self.background_roi.addScaleHandle([1, 0], [0, 1])  # Add a scale handle to allow resizing horizontally
        # self.background_roi.state={'pos': pg.Point(32.165493, 5262.965405), 'size': pg.Point(6.582061, 814.354380), 'angle': 0.0}
        # self.background_roi.lastState = {'pos': pg.Point(32.165493, 5262.965405), 'size': pg.Point(6.582061, 814.354380),'angle': 0.0}
        #self.background_roi.addScaleHandle([0, 1], [1, 0])  # Add a scale handle to allow resizing vertically

        self.video_view.addItem(self.background_roi)
        self.roi_button.setStyleSheet("background-color: green")

    def add_interest_roi(self):
        # Implement the logic to add ROI to the zone of interest
        # For demonstration purposes, let's create a rectangle at the bottom-right corner
        self.interest_roi = pg.ROI(pg.Point(38.938283, 2106.109990), pg.Point(17.818735, 950.761713), pen='g')#pg.ROI(pg.Point(39.400064, 1872.268849), pg.Point(23.514035, 1048.195522), pen='g'){'pos': Point(38.938283, 2106.109990), 'size': Point(17.818735, 950.761713), 'angle': 0.0}
        self.interest_roi.addScaleHandle([1, 0], [0, 1])  # Add a scale handle to allow resizing horizontally
        # self.interest_roi.state={'pos': pg.Point(39.400064, 1872.268849), 'size': pg.Point(23.514035, 1048.195522), 'angle': 0.0}
        # self.interest_roi.lastState = {'pos': pg.Point(39.400064, 1872.268849), 'size': pg.Point(23.514035, 1048.195522),'angle': 0.0}
        # self.interest_roi.addScaleHandle([0, 1], [1, 0])  # Add a scale handle to allow resizing vertically

        self.video_view.addItem(self.interest_roi)
        self.roi_button2.setStyleSheet("background-color: green")

    def calculate(self):

        if self.image_data is not None and self.image_data_filt is not None:#self.parent.mode_combo.currentIndex()==0 or self.parent.comparing:

            # Implement the logic to calculate the contrast to noise ratio
            if self.interest_roi is not None and self.background_roi is not None:#{'pos': Point(58.487016, 5204.505120), 'size': Point(6.582061, 814.354380), 'angle': 0.0}
                # Get the pixel values inside the ROIs
                interest_roi_pixels = self.interest_roi.getArrayRegion(self.image_data, self.image_view)
                background_roi_pixels = self.background_roi.getArrayRegion(self.image_data, self.image_view)

                # Calculate the contrast to noise ratio
                cnr_value = (np.mean(interest_roi_pixels) - np.mean(background_roi_pixels)) / np.std(background_roi_pixels)

                # Get the pixel values inside the ROIs
                interest_roi_pixels_filt = self.interest_roi.getArrayRegion(self.image_data_filt, self.image_view)
                background_roi_pixels_filt = self.background_roi.getArrayRegion(self.image_data_filt, self.image_view)

                # Calculate the contrast to noise ratio
                cnr_value_filt = (np.mean(interest_roi_pixels_filt) - np.mean(background_roi_pixels_filt)) / np.std(background_roi_pixels_filt)
                self.parent.cnr_label_filt.setText("CNR: {:.2f}".format(cnr_value_filt))
                self.parent.cnr_label.setText("CNR: {:.2f}".format(cnr_value))

                print(f"Contraste al Ruido: {cnr_value}")
            else:
                print("Por favor, seleccione ambas regiones de interés (ROI).")
        else:

            # Implement the logic to calculate the contrast to noise ratio
            if self.interest_roi is not None and self.background_roi is not None:


                # Get the pixel values inside the ROIs
                interest_roi_pixels_filt = self.interest_roi.getArrayRegion(self.image_data_filt, self.image_view)
                background_roi_pixels_filt = self.background_roi.getArrayRegion(self.image_data_filt, self.image_view)

                # Calculate the contrast to noise ratio
                cnr_value_filt = (np.mean(interest_roi_pixels_filt) - np.mean(background_roi_pixels_filt)) / np.std(
                    background_roi_pixels_filt)
                self.parent.cnr_label_filt.setText("CNR: {:.2f}".format(cnr_value_filt))

class STD_ROIDialog(QtWidgets.QDialog):
    def __init__(self,multiple_shot, image_data=None, image_data_filt=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Calculo de STD')
        self.image_data = image_data
        self.image_data_filt=image_data_filt
        self.parent=parent

        layout = QtWidgets.QVBoxLayout()
        self.video_view = pg.PlotWidget()
        self.video_view.hideAxis('bottom')
        self.video_view.hideAxis('left')
        self.video_view.setAspectLocked(False)
        # Item para mostrar image data



        self.image_view = pg.ImageItem()
        self.video_view.addItem(self.image_view)
        self.video_view.invertY()
        #self.image_view.setAspectLocked(False)
        if multiple_shot or  self.parent.checkbox2.isChecked():
            self.image_view.setImage(self.image_data_filt)
        else:
            self.image_view.setImage(self.image_data)

        self.roi_button = QtGui.QPushButton('ROI')
        self.roi_button.clicked.connect(self.add_std_roi)


        self.calculate_button = QtGui.QPushButton('Calculate')
        self.calculate_button.clicked.connect(self.calculate)

        layout.addWidget(self.video_view)
        layout.addWidget(self.roi_button)

        layout.addWidget(self.calculate_button)

        self.setLayout(layout)

        self.std_roi = self.parent.std_roi #state {'pos': Point(39.400064, 1755.348278), 'size': Point(23.052254, 872.814665), 'angle': 0.0}
        if not self.std_roi== None:
            self.roi_button.setStyleSheet("background-color: green")
            self.video_view.addItem(self.std_roi)




    def add_std_roi(self):
        # Implement the logic to add ROI to the zone of interest
        # For demonstration purposes, let's create a rectangle at the bottom-right corner
        self.std_roi = pg.ROI(pg.Point(38.938283, 2106.109990), pg.Point(17.818735, 950.761713), pen='g')#pg.ROI(pg.Point(39.400064, 1872.268849), pg.Point(23.514035, 1048.195522), pen='g')
        self.std_roi.addScaleHandle([1, 0], [0, 1])  # Add a scale handle to allow resizing horizontally
        # self.std_roi.state={'pos': pg.Point(39.400064, 1872.268849), 'size': pg.Point(23.514035, 1048.195522), 'angle': 0.0}
        # self.interest_roi.addScaleHandle([0, 1], [1, 0])  # Add a scale handle to allow resizing vertically

        self.video_view.addItem(self.std_roi)
        self.roi_button.setStyleSheet("background-color: green")

    def calculate(self):

        if self.image_data is not None and self.image_data_filt is not None:#self.parent.mode_combo.currentIndex()==0 or self.parent.comparing:

            # Implement the logic to calculate the contrast to noise ratio
            if self.std_roi is not None:
                # Get the pixel values inside the ROIs
                std_roi_pixels = self.std_roi.getArrayRegion(self.image_data, self.image_view)


                # Calculate the contrast to noise ratio
                std_value = np.std(std_roi_pixels)

                # Get the pixel values inside the ROIs
                std_roi_pixels_filt = self.std_roi.getArrayRegion(self.image_data_filt, self.image_view)


                # Calculate the contrast to noise ratio
                std_value_filt = np.std(std_roi_pixels_filt)
                self.parent.std_label_filt.setText("STD: {:.2f}".format(std_value_filt))
                self.parent.std_label.setText("STD: {:.2f}".format(std_value))


            else:
                print("Por favor, seleccione ambas regiones de interés (ROI).")
        else:

            # Implement the logic to calculate the contrast to noise ratio
            if self.std_roi is not None:


                # Get the pixel values inside the ROIs
                std_roi_pixels_filt = self.std_roi.getArrayRegion(self.image_data_filt, self.image_view)


                # Calculate the contrast to noise ratio
                std_value_filt = np.std(std_roi_pixels_filt)
                self.parent.std_label_filt.setText("STD: {:.2f}".format(std_value_filt))

class SimpleAcquisitionWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        # Configurar la ventana de adquisiciones múltiples
        self.setWindowTitle('Adquisicion disparo simple')
        self.setGeometry(100, 100, 800, 600)

        # Create a vertical layout for main window
        main_layout = QtGui.QVBoxLayout()
        # Crear un botón para agregar adquisiciones
        self.calculate_result_button = QtGui.QPushButton('Resultados')
        self.calculate_result_button.clicked.connect(self.calculate_result)

        # Create a horizontal layout for buttons
        button_layout = QtGui.QHBoxLayout()
        button_layout.addWidget(self.calculate_result_button)

        # Create a QWidget to hold the horizontal button layout
        self.button_widget = QtGui.QWidget()
        self.button_widget.setLayout(button_layout)


        # Estructura de la multiples adq

        self.acquisitions_layout = QtGui.QGridLayout()
        # Create a QWidget to hold the horizontal button layout
        self.acq_widget = QtGui.QWidget()
        self.acq_widget.setLayout(self.acquisitions_layout)

        # adding widget to main layout

        main_layout.addWidget(self.button_widget)
        main_layout.addWidget(self.acq_widget)
        self.setLayout(main_layout)

        # Lista para almacenar las adquisiciones
        self.acquisitions = []
        self.img_list = []
        self.n_acquisitions = 0
        # Lista para almacenar las subbandas
        self.subbands = []

        # valores filtro por defecto
        self.n_subbands = 1
        self.central_frequency = 3.5e6
        self.total_bandwidth = 3e6
        self.overlap_frequency = 0

        # Conecta la señal rejected (evento de cierre) al método close_and_delete
        #self.rejected.connect(self.close_and_delete)



        self.setModal(False)  # permite poder usar la main window, hay que ponerlo antes de mostrar
        self.add_acquisition()
        self.show()
    def close_and_delete(self):
        # Realiza cualquier limpieza necesaria antes de cerrar
        self.deleteLater()
    def add_acquisition(self):

        #self.acquisitions= self.parent.filt_sub_bscans
        self.n_subbands=self.parent.n_subbands
        # información necesario para aplicar scanConvert
        infoConvert = [self.parent.dset.dB_range, self.parent.dset.frame_ny, self.parent.dset.frame_nx, self.parent.dset.frames_val,
                       self.parent.dset.ix_test, self.parent.dset.w_test]


        for i in range(self.n_subbands):

            sub_bscan=copy.copy(self.parent.filt_sub_bscans[:,:,i]) #bscan envolvente de cada subfrecuencia
            # Actualizamos frame
            new_frame = np.ones((infoConvert[1], infoConvert[2]), dtype='h') * (-infoConvert[0].astype('h'))  # frame compuesto
            ScanConvert_BilinealFrame(infoConvert[1], infoConvert[2], infoConvert[3], new_frame, sub_bscan, infoConvert[4], infoConvert[5])
            # creo widget nuevo
            acq_filt_widget = acq_Widget(infoConvert, sub_bscan, new_frame,i, self)


            if i < 3:
                # acq_widget=QtGui.QWidget()
                # acq_widget.setLayout(acq_layout)
                self.acquisitions_layout.addWidget(acq_filt_widget, 0, i)
            elif i < 6:
                # acq_widget = QtGui.QWidget()
                # acq_widget.setLayout(acq_layout)
                self.acquisitions_layout.addWidget(acq_filt_widget, 1, i - 3)

            elif i < 9:
                # acq_widget = QtGui.QWidget()
                # acq_widget.setLayout(acq_layout)
                self.acquisitions_layout.addWidget(acq_filt_widget, 2, i - 6)

            # Agregar la adquisición a la lista
            if i == 0:
                envolvente=copy.deepcopy(sub_bscan)
                self.acquisitions = np.expand_dims(envolvente, axis=2)
                self.img_list.append(acq_filt_widget)
            else:
                envolvente = np.expand_dims(sub_bscan, axis=2)
                self.acquisitions = np.append(self.acquisitions, envolvente, axis=2)
                self.img_list.append(acq_filt_widget)


    def calculate_result(self):

        num_samples, num_lines, n_acq = self.acquisitions.shape

        # Inicializar la imagen compuesta
        compounded_image = np.zeros((num_samples, num_lines))
        # información necesario para aplicar scanConvert
        infoConvert = [self.parent.dset.dB_range, self.parent.dset.frame_ny, self.parent.dset.frame_nx,
                       self.parent.dset.frames_val,
                       self.parent.dset.ix_test, self.parent.dset.w_test]

        # Al hacer click este boton habilito la opcion para que se calcule el resultado y se muestre en la app principal
        for i in range(n_acq):
            envolvente= self.acquisitions[:,:,i]
            # Acumular la sub-banda filtrada en la imagen compuesta
            compounded_image += envolvente

        # Normalizar la imagen compuesta

        compounded_image /= n_acq

        # self.compounded_bscan=compounded_image
        self.filt_frame = np.ones((infoConvert[1], infoConvert[2]), dtype='h') * (-infoConvert[0].astype('h'))  # frame compuesto
        ScanConvert_BilinealFrame(infoConvert[1], infoConvert[2], infoConvert[3], self.filt_frame , compounded_image, infoConvert[4], infoConvert[5])
        self.compounded_bscan = self.filt_frame
        self.parent.bscan_filtered[:, :, 0]=compounded_image


        self.parent.img_filtered.setImage(self.filt_frame)

class MultiAcquisitionWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__()
        self.parent=parent
        # Configurar la ventana de adquisiciones múltiples
        self.setWindowTitle('Adquisicion múltiples disparos')
        self.setGeometry(100, 100, 800, 600)

        # Crear un botón para agregar adquisiciones
        self.add_acquisition_button = QtGui.QPushButton('Agregar Adquisición')
        self.add_acquisition_button.clicked.connect(self.add_acquisition)
        self.add_acquisition_button.setEnabled(False)
        # Crear un botón para agregar adquisiciones
        self.calculate_result_button = QtGui.QPushButton('Resultados')
        self.calculate_result_button.clicked.connect(self.calculate_result)
        self.calculate_result_button.setEnabled(False)
        # Create button to open the configuration dialog
        self.config_button = QtGui.QPushButton("Configurar Filtros")
        # Etiqueta para mostrar por pantalla las subbandas usadas
        self.subbands_label = QtGui.QLabel('Subbandas de frecuencia usadas: ')
        self.config_button.clicked.connect(self.open_config_dialog)
        # Conecta la señal rejected (evento de cierre) al método close_and_delete
        #self.rejected.connect(self.close_and_delete)



        # Create a vertical layout for main window
        main_layout = QtGui.QVBoxLayout()

        # Create a horizontal layout for buttons
        button_layout = QtGui.QHBoxLayout()
        button_layout.addWidget(self.add_acquisition_button)
        button_layout.addWidget(self.calculate_result_button)
        button_layout.addWidget(self.config_button)
        button_layout.addWidget(self.subbands_label)
        # Create a QWidget to hold the horizontal button layout
        self.button_widget = QtGui.QWidget()
        self.button_widget.setLayout(button_layout)

        # Estructura de la multiples adq

        self.acquisitions_layout = QtGui.QGridLayout()
        # Create a QWidget to hold the horizontal button layout
        self.acq_widget = QtGui.QWidget()
        self.acq_widget.setLayout(self.acquisitions_layout)

        # adding widget to main layout

        main_layout.addWidget(self.button_widget)
        main_layout.addWidget(self.acq_widget)
        self.setLayout(main_layout)

        # Lista para almacenar las adquisiciones
        self.acquisitions = []
        self.img_list = []
        self.n_acquisitions=0
        # Lista para almacenar las subbandas
        self.subbands = []

        # valores filtro por defecto
        self.n_subbands = 1
        self.central_frequency = 3.5e6
        self.total_bandwidth = 3e6
        self.overlap_frequency = 0

        # Create an instance of FrequencyCompoundingUI
        self.config_dialog = FrequencyCompoundingUI()

        self.setModal(False)  # permite poder usar la main window, hay que ponerlo antes de mostrar
        self.show()



    def close_and_delete(self):
        # Realiza cualquier limpieza necesaria antes de cerrar
        self.deleteLater()
    def add_acquisition(self):


        #Cargamos una adquisición
        video_file, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open Video', '', 'Video Files (*.BIN)')
        if video_file:
            # Leo el archivo y genero un objeto 'BinFile' que contiene los datos crudos tal cual se leen del archivo
            bfile = BinFile(video_file)
            # A partir del objeto 'BinFile' genero un objeto 'Dataset' que ya tiene los datos en formato 'cubo de datos'
            # y contiene las matrices de coordenadas de los píxeles para poder pintar
            self.dset = Dataset(bfile, resample=0, med_kernel=0, avg_kernel=0)

            fs = 40e6  # Frecuencia de muestreo en Hz

            [start, end] = self.subbands[self.n_acquisitions]
            b = signal.firwin(128, [start, end], pass_zero='bandpass', fs=fs, scale=True)
            a = 1
            # Filtrar la matriz cruda B-scan con el filtro paso banda
            filtered_b_scan = signal.filtfilt(b, a, self.dset.bscan[:, :, 0], axis=0)
            envolvente = signal.hilbert(filtered_b_scan, axis=0)
            envolvente = 20 * np.log10(np.abs(envolvente) / 3276.7)  # TAMAÑO INT16 pasamos a dB
            envolvente[envolvente < -self.dset.dB_range] = -self.dset.dB_range
            #simulo que solo tiene una imagen el resto no me interesa
            self.dset.bscan[:,:,0] = envolvente

            self.dset.ScanConvert(raw=False)  # como bscan y raw son iguales trabajo con uno para filtrar y el otro pinto para comparar
            #información necesario para aplicar scanConvert
            infoConvert=[self.dset.dB_range,self.dset.frame_ny, self.dset.frame_nx, self.dset.frames_val, self.dset.ix_test, self.dset.w_test]

            #creo widget nuevo
            acq_filt_widget = acq_Widget(infoConvert, self.dset.bscan[:,:,0], self.dset.frames[:,:,0], self.n_acquisitions, self)



            if self.n_acquisitions<3:
                # acq_widget=QtGui.QWidget()
                # acq_widget.setLayout(acq_layout)
                self.acquisitions_layout.addWidget(acq_filt_widget,0,  self.n_acquisitions)
            elif self.n_acquisitions<6:
                # acq_widget = QtGui.QWidget()
                # acq_widget.setLayout(acq_layout)
                self.acquisitions_layout.addWidget(acq_filt_widget, 1, self.n_acquisitions-3)

            elif self.n_acquisitions < 9:
                # acq_widget = QtGui.QWidget()
                # acq_widget.setLayout(acq_layout)
                self.acquisitions_layout.addWidget(acq_filt_widget, 2, self.n_acquisitions - 6)


            # Agregar la adquisición a la lista
            if self.n_acquisitions == 0:
                self.acquisitions=np.expand_dims(envolvente, axis=2)
                self.img_list.append(acq_filt_widget)
            else:
                envolvente=np.expand_dims(envolvente, axis=2)
                self.acquisitions=np.append(self.acquisitions,envolvente, axis=2)
                self.img_list.append(acq_filt_widget)
            self.n_acquisitions+=1
            # Habilitar el botón de cálculo cuando al menos haya una adquisición

            if self.n_acquisitions >= len(self.subbands):
                self.add_acquisition_button.setEnabled(False)
                self.calculate_result_button.setEnabled(True)
            del bfile

    def calculate_result(self):

        num_samples, num_lines, n_acq = self.acquisitions.shape

        # Inicializar la imagen compuesta
        compounded_image = np.zeros((num_samples, num_lines))

        # Al hacer click este boton habilito la opcion para que se calcule el resultado y se muestre en la app principal
        for i in range(self.n_acquisitions):
            envolvente= self.acquisitions[:,:,i]
            # Acumular la sub-banda filtrada en la imagen compuesta
            compounded_image += envolvente

        # Normalizar la imagen compuesta

        compounded_image /= self.n_acquisitions

        # self.compounded_bscan=compounded_image
        self.filt_frame = np.ones((self.dset.frame_ny, self.dset.frame_nx), dtype='h') * (-self.dset.dB_range.astype('h')) #frame compuesto
        ScanConvert_BilinealFrame(self.dset.frame_ny, self.dset.frame_nx, self.dset.frames_val, self.filt_frame, compounded_image, self.dset.ix_test, self.dset.w_test)

        self.compounded_bscan = self.filt_frame
        self.parent.img_filtered.setImage(self.filt_frame)
        self.parent.subbands_string=self.subbands_string
        self.parent.subbands_label.setText('Subbandas de frecuencia usadas: ' + self.subbands_string)





    def open_config_dialog(self):

        # Load the configuration values before showing the dialog
        self.config_dialog.subbands_spinbox.setValue(self.n_subbands)
        self.config_dialog.central_frequency_spinbox.setValue(self.central_frequency/10**6)
        self.config_dialog.bandwidth_spinbox.setValue(self.total_bandwidth/10**6)
        self.config_dialog.overlap_spinbox.setValue(self.overlap_frequency/10**6)

        if self.config_dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Apply the settings from the dialog
            self.n_subbands = int(self.config_dialog.subbands_spinbox.value())
            self.central_frequency = self.config_dialog.central_frequency_spinbox.value() * 10**6
            self.total_bandwidth = self.config_dialog.bandwidth_spinbox.value()* 10**6
            self.overlap_frequency=self.config_dialog.overlap_spinbox.value()* 10**6

            self.subbands = calculate_subbands(self.total_bandwidth, self.n_subbands, self.overlap_frequency,
                                          self.central_frequency)
            self.subbands_string = ",".join(f"({start / 1e6:.2f}-{end / 1e6:.2f}) MHz" for start, end in self.subbands)
            print(self.subbands_string)
            self.subbands_label.setText('Subbandas de frecuencia usadas: ' + self.subbands_string)
            self.add_acquisition_button.setEnabled(True)

    # def changeValue(self, n_acquisition):
    #     #Aqui en funcion del slider que se active se modifica el bscan que corresponda y se actualiza la imagen
    #     value=self.slider.value()
    #     new_bscan=self.acquisitions[n_acquisition]*value
    #     #Actualizamos frame
    #     new_frame = np.ones((self.dset.frame_ny, self.dset.frame_nx), dtype='h') * (
    #         -self.dset.dB_range.astype('h'))  # frame compuesto
    #     ScanConvert_BilinealFrame(self.dset.frame_ny, self.dset.frame_nx, self.dset.frames_val, new_frame,new_bscan, self.dset.ix_test, self.dset.w_test)
    #
    #     self.img_list[n_acquisition].setImage[new_frame]


class SpatialWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__()
        self.parent=parent
        # Configurar la ventana de adquisiciones múltiples
        self.setWindowTitle('Adquisicion múltiples disparos')
        self.setGeometry(100, 100, 800, 600)

        # Crear un botón para agregar adquisiciones
        self.add_acquisition_button = QtGui.QPushButton('Agregar Adquisición')
        self.add_acquisition_button.clicked.connect(self.add_acquisition)
        self.add_acquisition_button.setEnabled(False)
        # Crear un botón para agregar adquisiciones
        self.calculate_result_button = QtGui.QPushButton('Resultados')
        self.calculate_result_button.clicked.connect(self.calculate_result)
        self.calculate_result_button.setEnabled(False)

        self.spatial_config_dialog_button = QtGui.QPushButton('Configurar filt. espaciales')
        self.spatial_config_dialog_button.clicked.connect(self.open_config_dialog)
        # Conecta la señal rejected (evento de cierre) al método close_and_delete
        #self.rejected.connect(self.close_and_delete)



        # Create a vertical layout for main window
        main_layout = QtGui.QVBoxLayout()

        # Create a horizontal layout for buttons
        button_layout = QtGui.QHBoxLayout()
        button_layout.addWidget(self.add_acquisition_button)
        button_layout.addWidget(self.calculate_result_button)
        button_layout.addWidget(self.spatial_config_dialog_button)

        # Create a QWidget to hold the horizontal button layout
        self.button_widget = QtGui.QWidget()
        self.button_widget.setLayout(button_layout)

        # Estructura de la multiples adq

        self.acquisitions_layout = QtGui.QGridLayout()
        # Create a QWidget to hold the horizontal button layout
        self.acq_widget = QtGui.QWidget()
        self.acq_widget.setLayout(self.acquisitions_layout)



        # adding widget to main layout

        main_layout.addWidget(self.button_widget)
        main_layout.addWidget(self.acq_widget)
        self.setLayout(main_layout)

        # Lista para almacenar las adquisiciones
        self.acquisitions = []
        self.img_list = []
        self.elem_list=[]



        # valores filtro por defecto
        self.n_acquisitions = 0
        self.n_steps=5
        self.aperture = 48
        self.n_elements = 64
        self.last_el = 16
        self.first_el = 1
        self.pitch=0.34
        self.deltaMax = 15*0.34  # valor en mm
        # self.c_x=self.deltaMax/2

        # Create an instance of SpatialCompoundingUI
        self.spatial_config_dialog = SpatialCompoundingUI()

        self.setModal(False)  # permite poder usar la main window, hay que ponerlo antes de mostrar
        self.show()




    def close_and_delete(self):
        # Realiza cualquier limpieza necesaria antes de cerrar
        self.deleteLater()
    def add_acquisition(self):


        #Cargamos una adquisición
        video_file, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open Video', '', 'Video Files (*.BIN)')
        if video_file:
            # Leo el archivo y genero un objeto 'BinFile' que contiene los datos crudos tal cual se leen del archivo
            bfile = BinFile(video_file)
            # A partir del objeto 'BinFile' genero un objeto 'Dataset' que ya tiene los datos en formato 'cubo de datos'
            # y contiene las matrices de coordenadas de los píxeles para poder pintar
            self.dset = Dataset(bfile, resample=0, med_kernel=0, avg_kernel=0)
            elem_num=int(self.dset.filename.split('.')[0].split('/')[-1])

            fs = 40e6  # Frecuencia de muestreo en Hz
            # Diseñar el filtro paso banda

            b = signal.firwin(128, [1e6, 4e6], pass_zero='bandpass', fs=fs, scale=True)
            a = 1
            # Filtrar la matriz cruda B-scan con el filtro paso banda
            filtered_b_scan = signal.filtfilt(b, a, self.dset.bscan[:, :, 0], axis=0)
            envolvente = signal.hilbert(filtered_b_scan, axis=0)
            envolvente = 20 * np.log10(np.abs(envolvente) / 3276.7)  # TAMAÑO INT16 pasamos a dB
            envolvente[envolvente < -self.dset.dB_range] = -self.dset.dB_range
            #simulo que solo tiene una imagen el resto no me interesa
            self.dset.bscan[:,:,0] = envolvente

            self.dset.ScanConvert(raw=False)  # como bscan y raw son iguales trabajo con uno para filtrar y el otro pinto para comparar
            #información necesario para aplicar scanConvert
            infoConvert=[self.dset.dB_range,self.dset.frame_ny, self.dset.frame_nx, self.dset.frames_val, self.dset.ix_test, self.dset.w_test]

            #creo widget nuevo
            acq_filt_widget = acq_Widget(infoConvert, self.dset.bscan[:,:,0], self.dset.frames[:,:,0], self.n_acquisitions, self)



            if self.n_acquisitions<3:
                # acq_widget=QtGui.QWidget()
                # acq_widget.setLayout(acq_layout)
                self.acquisitions_layout.addWidget(acq_filt_widget,0,  self.n_acquisitions)
            elif self.n_acquisitions<6:
                # acq_widget = QtGui.QWidget()
                # acq_widget.setLayout(acq_layout)
                self.acquisitions_layout.addWidget(acq_filt_widget, 1, self.n_acquisitions-3)

            elif self.n_acquisitions < 9:
                # acq_widget = QtGui.QWidget()
                # acq_widget.setLayout(acq_layout)
                self.acquisitions_layout.addWidget(acq_filt_widget, 2, self.n_acquisitions - 6)


            # Agregar la adquisición a la lista
            if self.n_acquisitions == 0:
                self.acquisitions=np.expand_dims(self.dset.frames[:,:,0], axis=2)
                self.img_list.append(acq_filt_widget)
                self.elem_list.append(elem_num)
            else:
                frame=np.expand_dims(self.dset.frames[:,:,0], axis=2)
                self.acquisitions=np.append(self.acquisitions,frame, axis=2)
                self.img_list.append(acq_filt_widget)
                self.elem_list.append(elem_num)
            self.n_acquisitions+=1
            # Habilitar el botón de cálculo cuando al menos haya una adquisición

            if self.n_acquisitions >= self.n_steps:
                self.add_acquisition_button.setEnabled(False)
                self.calculate_result_button.setEnabled(True)
            del bfile

    def calculate_result(self):

        #SPATIAL COMP
        #final_frame=np.zeros((self.dset.frames.shape[0],self.dset.frames.shape[1]))
        print(self.dset.px2mm)
        frame_result=np.zeros((self.dset.frames.shape[0],self.dset.frames.shape[1]+np.ceil(self.deltaMax/self.dset.px2mm).astype(int)))#* (-self.dset.dB_range.astype('h')) Añado uno para tener en cuenta el pixel de los dos lados
        # se calcula nuevo centro de imagen en px
        c_1 = int(self.dset.cx)#np.ceil(frame_result.shape[1]/ 2).astype(int)
        c_x=c_1+np.round((self.deltaMax/2)/self.dset.px2mm).astype(int)
        frame_result_prom = np.zeros_like(frame_result) #matriz donde guanto los cuantas veces se has pintado los pixeles para calcular promedio
        frame_result_val = np.zeros((self.dset.frames.shape[0],self.dset.frames.shape[1]+np.ceil(self.deltaMax/self.dset.px2mm).astype(int)), dtype=bool)
        frame_result_val[:,c_x-int(self.dset.cx):c_x+int(self.dset.cx)+1]=self.dset.frames_val



        #num_samples, num_lines, n_acq = self.acquisitions.shape

        # Inicializar la imagen compuesta
        #compounded_image = np.zeros((num_samples, num_lines))

        # Al hacer click este boton habilito la opcion para que se calcule el resultado y se muestre en la app principal
        for i in range(self.n_acquisitions):
            frame= self.acquisitions[:,:,i]
            elem=self.elem_list[i]
            # calculo el centro de la imagen a sumar en pixeles
            c_im= np.ceil(c_1 + (elem-1)*self.pitch/self.dset.px2mm).astype(int)#np.ceil(c_x - (((self.last_el-self.first_el)/2)-elem)*self.pitch/self.dset.px2mm).astype(int)
            frame_result[:, c_im - int(self.dset.cx): c_im + int(self.dset.cx)+1]=(frame_result[:, c_im - int(self.dset.cx): c_im + int(self.dset.cx)+1]+(frame+self.dset.dB_range))#*frame_result_val
            frame_result=frame_result*frame_result_val

            frame_result_prom[:, c_im - int(self.dset.cx): c_im + int(self.dset.cx)+1]+=np.ones_like(frame)*self.dset.frames_val
            frame_result_prom = frame_result_prom * frame_result_val


        # Normalizar la imagen compuesta

        frame_result = np.divide(frame_result, frame_result_prom, out=np.zeros_like(frame_result), where=frame_result_prom!=0)
        frame_result = frame_result-self.dset.dB_range
        frame_result[frame_result < -self.dset.dB_range] = -self.dset.dB_range

        self.compounded_image=frame_result
        # self.filt_frame = np.ones((self.dset.frame_ny, self.dset.frame_nx), dtype='h') * (-self.dset.dB_range.astype('h')) #frame compuesto
        # ScanConvert_BilinealFrame(self.dset.frame_ny, self.dset.frame_nx, self.dset.frames_val, self.filt_frame, compounded_image, self.dset.ix_test, self.dset.w_test)

        self.parent.img_filtered.setImage(frame_result)
        # self.parent.subbands_string=self.subbands_string
        # self.parent.subbands_label.setText('Subbandas de frecuencia usadas: ' + self.subbands_string)





    def open_config_dialog(self):

        # Load the configuration values before showing the dialog
        self.spatial_config_dialog.n_step_spinBox.setValue(self.n_steps)
        self.spatial_config_dialog.aperture_spinBox.setValue(self.aperture)
        self.spatial_config_dialog.n_elements_spinBox.setValue(self.n_elements)

        if self.spatial_config_dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Apply the settings from the dialog
            self.n_steps = int(self.spatial_config_dialog.n_step_spinBox.value())
            self.aperture = self.spatial_config_dialog.aperture_spinBox.value()
            self.n_elements = self.spatial_config_dialog.n_elements_spinBox.value()
            self.pitch = self.spatial_config_dialog.pitch.value()
            self.last_el = self.spatial_config_dialog.last_spinbox.value()
            self.first_el = self.spatial_config_dialog.first_spinbox.value()
            self.deltaMax = (self.last_el -self.first_el) * self.pitch  # valor en mm


            self.add_acquisition_button.setEnabled(True)

    # def changeValue(self, n_acquisition):
    #     #Aqui en funcion del slider que se active se modifica el bscan que corresponda y se actualiza la imagen
    #     value=self.slider.value()
    #     new_bscan=self.acquisitions[n_acquisition]*value
    #     #Actualizamos frame
    #     new_frame = np.ones((self.dset.frame_ny, self.dset.frame_nx), dtype='h') * (
    #         -self.dset.dB_range.astype('h'))  # frame compuesto
    #     ScanConvert_BilinealFrame(self.dset.frame_ny, self.dset.frame_nx, self.dset.frames_val, new_frame,new_bscan, self.dset.ix_test, self.dset.w_test)
    #
    #     self.img_list[n_acquisition].setImage[new_frame]


class acq_Widget(QtGui.QWidget):
    def __init__(self, infoConvert, bscan, image, index, parent):
        super().__init__()
        self.parent=parent
        self.index=index
        self.infoConvert=infoConvert
        self.bscan=bscan
        self.frame=image

        acq_layout = QtGui.QHBoxLayout()
        # Una vez se haya cargado el frame se muestra y se añade a la vista
        # acquisition_label = QtGui.QLabel('Adquisición #{}'.format(dset.filename))
        self.acq_view = pg.PlotWidget()
        self.acq_view.hideAxis('bottom')
        self.acq_view.hideAxis('left')
        self.acq_view.setAspectLocked(True)
        # Item para mostrar image data
        self.img_acq = pg.ImageItem()
        self.img_acq.setImage(self.frame)
        self.acq_view.addItem(self.img_acq)
        self.acq_view.invertY()
        # Crear una etiqueta emergente para mostrar las coordenadas y el valor del píxel

        self.tooltip_label = QtGui.QLabel(self)
        self.tooltip_label.setStyleSheet("QLabel { background-color : white; color : black; }")
        self.tooltip_label.setWindowFlags(QtCore.Qt.ToolTip)
        self.tooltip_label.setMargin(5)

        # Añado slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.slider.setMinimum(100)  # Valor mínimo: 1.0
        self.slider.setMaximum(300)  # Valor máximo: 3.0
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self.update_image_saturation)
        #Creo una ventana emergente para mostrar el valor del slider cuando el cursor se pone sobre el
        # tooltip_window = SliderValueWindow(self.slider)

        # añadimos elementos al layout
        acq_layout.addWidget(self.slider)
        acq_layout.addWidget(self.acq_view)

        self.setLayout(acq_layout)
        self.mousePressEvent = self.mouse_click

    def mouse_click(self, event):
        pos = event.pos()
        # Convertir la posición de la escena a coordenadas de la imagen
        pos_img = self.img_acq.mapFromScene(pos)
        x, y = int(pos_img.x()), int(pos_img.y())

        # Obtener el valor del píxel en las coordenadas del clic
        pixel_value = self.img_acq.image[y, x]
        coordinates = f"Coordenadas: ({x}, {y})"
        pixel_info = f"Valor del píxel: {pixel_value}"

        # Mostrar el widget emergente cerca del cursor
        tooltip_text = f"{coordinates}\n{pixel_info}"
        self.tooltip_label.setText(tooltip_text)
        self.tooltip_label.move(QtGui.QApplication.desktop().cursor().pos() + QtCore.QPoint(10, 10))
        self.tooltip_label.show()

        # Ocultar el widget emergente después de un tiempo
        QtCore.QTimer.singleShot(2000, self.tooltip_label.hide)

    def update_image_saturation(self):
        # Obtener el valor del slider y calcular la saturación
        saturation = self.slider.value() / 100
        # Mostrar el valor en un tooltip
        self.setToolTip(f'Valor: {saturation:.2f}')

        # Aplicar la saturación a la imagen y actualizarla
        new_bscan = ((self.bscan + self.infoConvert[0])*saturation) - self.infoConvert[0]
        new_bscan = new_bscan * (new_bscan <= 0) + 0 * (new_bscan > 0) #saturo los valores a 0 (rango dinamico de -48 a 0)
        # Actualizamos frame
        new_frame = np.ones((self.infoConvert[1], self.infoConvert[2]), dtype='h') * (-self.infoConvert[0].astype('h'))  # frame compuesto
        ScanConvert_BilinealFrame(self.infoConvert[1], self.infoConvert[2], self.infoConvert[3], new_frame, new_bscan, self.infoConvert[4], self.infoConvert[5])

        self.frame=copy.copy(new_frame)

        #actualizamos bscan en el padre
        self.parent.acquisitions[:, :, self.index]=copy.copy(new_bscan)

        self.img_acq.setImage(self.frame)

        del new_frame
        del new_bscan

        #self.parent.app.processEvents()



# class SliderValueWindow(QtGui.QWidget):
#     def __init__(self, slider):
#         super().__init__()
#         self.slider = slider
#         self.slider.valueChanged.connect(self.show_value_tooltip)
#
#     def show_value_tooltip(self):
#         # Obtener el valor actual del slider
#         value = self.slider.value() / 100.0  # Convertir el valor a decimal (por ejemplo, 1.0 a 3.0)
#
#         # Mostrar el valor en un tooltip
#         self.setToolTip(f'Valor: {value:.2f}')


def FIR_linear_filter(signal_rf):
    # Definir los coeficientes del filtro FIR (ejemplo de filtro pasabajo)
    order = 80  # Orden del filtro
    cutoff = 0.2  # Frecuencia de corte relativa (debe estar entre 0 y 1)
    b = signal.firwin(order, cutoff)

    # Aplicar el filtro FIR a la señal de ultrasonido
    filtered_signal = signal.lfilter(b, 1, signal_rf, axis=0)

    # Normalizar la señal filtrada
    filtered_signal /= np.max(filtered_signal)

    return filtered_signal





if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    player = VideoPlayer(app)
    player.show()
    sys.exit(app.exec_())