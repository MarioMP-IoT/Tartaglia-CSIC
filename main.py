"""Script para proyecto Tartaglia"""

import sys
import cv2
import copy
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import time
from scipy import signal

from file_functions import *




class VideoPlayer(QtGui.QWidget):

    def __init__(self, app):
        super().__init__()
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

        self.video_view.addItem(self.img_video)
        self.video_view.invertY()



        self.filtered_image_view = pg.PlotWidget()
        self.filtered_image_view.hideAxis('bottom')
        self.filtered_image_view.hideAxis('left')
        self.filtered_image_view.invertY()
        self.filtered_image_view.setAspectLocked(True)
        # Item para mostrar image data
        self.img_filtered = pg.ImageItem()
        #self.img_filtered.setLevels((0, 255))

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
        #Etiqueta para mostrar por pantalla las subbandas usadas
        self.subbands_label = QtGui.QLabel('Subbandas de frecuencia usadas: ')

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
        layout.addWidget(self.video_label,1,0,1,1)


        layout.addWidget(self.video_view,2,0,2,1)
        # self.video_view.setMinimumHeight(350)  # Añado tamaño minimo a la figura principal

        layout.addWidget(self.checkbox1,0,1,1,1)
        layout.addWidget(self.checkbox2,2,1,1,1)
        layout.addWidget(self.subbands_label,1,1,1,1)

        layout.addWidget(self.filtered_image_view, 3, 1, 1, 2)
        #layout.addWidget(self.scroll,4,0,1,3)
        #layout.addWidget(self.button_widget,4,0,1,1)
        layout.addWidget(self.button_widget_filt, 4, 0, 1, 3)
        layout.addWidget(self.labels_result_widget,5,0,1,3)

        layout.addWidget(self.config_button, 0, 2, 1, 1)

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


        # # Connect checkbox signals to slots
        self.checkbox1.stateChanged.connect(self.update_images)
        self.checkbox2.stateChanged.connect(self.update_images)
        #self.checkbox3.stateChanged.connect(self.update_images)

        # Create an instance of FrequencyCompoundingUI
        self.config_dialog = FrequencyCompoundingUI()




        # Initialize variables
        self.video_file = ''
        self.video_capture = None
        self.video_frame = None

        self.roi_mask = None
        self.image=None

        self.prev_scroll_pos=0
        self.frame_pos=0
        #valores filtro por defecto
        self.n_subbands = 1
        self.central_frequency = 3.5e6
        self.total_bandwidth = 3e6
        self.overlap_frequency = 0



    def clearWindows(self):

        # Initialize variables
        self.video_file = ''
        self.video_capture = None
        self.video_frame = None

        self.prev_scroll_pos = 0
        self.frame_pos = 0



        # self.p0 = None
    def open_video_file(self):
        if self.video_capture is not None:
            self.clearWindows()

        "funcion para cargar los .bin"



        self.video_file, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open Video', '', 'Video Files (*.BIN)')
        if self.video_file:
            # Leo el archivo y genero un objeto 'BinFile' que contiene los datos crudos tal cual se leen del archivo
            self.bfile = BinFile(self.video_file)
            # A partir del objeto 'BinFile' genero un objeto 'Dataset' que ya tiene los datos en formato 'cubo de datos'
            # y contiene las matrices de coordenadas de los píxeles para poder pintar
            self.dset = Dataset(self.bfile, resample=0, med_kernel=0, avg_kernel=0)
            for i in range(self.dset.nimg):

                envolvente = signal.hilbert(self.dset.bscan[:,:,i], axis=0)
                envolvente = 20 * np.log10(np.abs(envolvente) / 3276.7)  # TAMAÑO INT16 pasamos a dB
                envolvente[envolvente< -self.dset.dB_range] = -self.dset.dB_range
                self.dset.bscan[:, :, i]=envolvente


            self.dset.ScanConvert(raw=False) #como bscan y raw son iguales trabajo con uno para filtrar y el otro pinto para comparar


            self.video_label.setText('Video: ' + self.video_file)
            # self.frames = []
            self.scroll.setRange(0, self.dset.nimg - 1)  # Añado rango del scroll en funcion del numero de imagenes
            self.scroll.setSliderPosition(0)


            self.update_images()

            self.first = True  # bandera para poder cargar imagen to track la primera vez
            self.video_openned = True
            self.scroll.show()

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
            self.update_images(only_filtered=True)

    def video_review(self):
        # getting current value of scroll bar
        frame_pos = self.scroll.value()
        self.frame_pos = frame_pos


        #llamo a la funcion de pintado para actualizar las imagenes
        self.update_images()


    def update_images(self, only_filtered=False):
        if not only_filtered:
            self.image=self.dset.frames[:,:,self.frame_pos]

            self.img_video.setImage(self.image)

        # if self.tracking:
        self.filter_image()
        self.dset.ScanConvert(filtered_frame=self.bscan_filtered)
        self.image_filtered=self.dset.filt_frame
        self.img_filtered.setImage(self.image_filtered, autoLevels=True)

        self.app.processEvents()

    def filter_image(self):
        # Copy the original image
        processed_image = self.dset.raw_bscan[:,:,self.frame_pos].copy()
        empty=False

        # Apply filters based on checkbox states
        if self.checkbox1.isChecked(): #frequency compounded
            print('AQUI VA FILTRO')
            #Calculamos las subbandas

            #subbands = [(2e6, 2.5e6),(2.5e6, 3e6), (3e6, 3.5e6),(3.5e6, 4e6)]

            fs = 40e6 # Frecuencia de muestreo en Hz
            subbands=calculate_subbands(self.total_bandwidth, self.n_subbands, self.overlap_frequency, self.central_frequency)
            subbands_string = ",".join(f"({start / 1e6:.2f}-{end / 1e6:.2f}) MHz" for start, end in subbands)
            print(subbands_string)
            self.subbands_label.setText('Subbandas de frecuencia usadas: '+ subbands_string)


            processed_image= self.frequency_compounded_imaging(processed_image, subbands,fs)


        elif self.checkbox2.isChecked():#spatial compounded
            print('AQUI VA FILTRO')

        # elif self.checkbox3.isChecked(): #FIR
        #     print('AQUI VA FILTRO')
        #     processed_image=FIR_linear_filter(processed_image)
        else:
            empty=True

        if empty:
            envolvente = signal.hilbert(processed_image, axis=0)
            envolvente = 20 * np.log10(np.abs(envolvente) / 3276.7)  # TAMAÑO INT16 pasamos a dB
            envolvente[envolvente < -self.dset.dB_range] = -self.dset.dB_range
            # change the processed image
            self.bscan_filtered = envolvente
        else:
            # change the processed image
            self.bscan_filtered=processed_image

    def frequency_compounded_imaging(self, b_scan, subbands, fs):
        num_subbands = len(subbands)
        num_samples, num_lines = b_scan.shape

        # Inicializar la imagen compuesta
        compounded_image = np.zeros((num_samples, num_lines))

        # Aplicar Frequency Compounded Imaging
        for start, end in subbands:
            # Diseñar el filtro paso banda
            
            b = signal.firwin(32, [start, end], pass_zero='bandpass', fs=fs, scale=True)
            a = 1
            # Filtrar la matriz cruda B-scan con el filtro paso banda
            filtered_b_scan = signal.filtfilt(b, a, b_scan, axis=0)
            envolvente = signal.hilbert(filtered_b_scan, axis=0)
            envolvente = 20 * np.log10(np.abs(envolvente)/3276.7)  # TAMAÑO INT16 pasamos a dB
            envolvente[envolvente < -self.dset.dB_range] = -self.dset.dB_range

            # Acumular la sub-banda filtrada en la imagen compuesta
            compounded_image += envolvente

        # Normalizar la imagen compuesta
        compounded_image /= num_subbands

        # compounded_image=(compounded_image)/np.min(compounded_image)*(self.dset.dB_range) -self.dset.dB_range#convierto a valores de -48 a 0

        return compounded_image



    def calculate_std(self):
        # calculate the standard deviation of the image data

        std_value_filt = np.std(self.bscan_filtered)

        std_value = np.std(self.dset.bscan[:,:,0])

        self.std_label.setText("STD: {:.2f}".format(std_value))
        self.std_label_filt.setText("STD: {:.2f}".format(std_value_filt))
        #print(f"Desviación Típica: {std_value}")

    def calculate_cnr(self):

        image_data_filt = self.bscan_filtered

        image_data = self.dset.bscan[:,:,0]

        # Create the ROI dialog and show it
        self.roi_dialog = ROIDialog(image_data,image_data_filt, self)
        self.roi_dialog.exec_()


    def open_roi_dialog(self,FILT=False):
        if FILT:
            image_data = self.bscan_filtered
        else:
            image_data = self.dset.bscan[:,:,0].copy()
        # Create the ROI dialog and show it
        self.roi_dialog = ROIDialog(image_data, self)
        self.roi_dialog.exec_()

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
        self.subbands_spinbox.setMaximum(10)
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

class ROIDialog(QtWidgets.QDialog):
    def __init__(self, image_data, image_data_filt, parent=None):
        super().__init__(parent)
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

        self.background_roi = None
        self.interest_roi = None

    def add_background_roi(self):
        # Implement the logic to add ROI to the background
        # For demonstration purposes, let's create a rectangle at the top-left corner
        self.background_roi = pg.ROI([50, 60], [70, 80], pen='r')
        self.background_roi.addScaleHandle([1, 0], [0, 1])  # Add a scale handle to allow resizing horizontally
        #self.background_roi.addScaleHandle([0, 1], [1, 0])  # Add a scale handle to allow resizing vertically

        self.video_view.addItem(self.background_roi)
        self.roi_button.setStyleSheet("background-color: green")

    def add_interest_roi(self):
        # Implement the logic to add ROI to the zone of interest
        # For demonstration purposes, let's create a rectangle at the bottom-right corner
        self.interest_roi = pg.ROI([50, 60], [70, 80], pen='g')
        self.interest_roi.addScaleHandle([1, 0], [0, 1])  # Add a scale handle to allow resizing horizontally
        # self.interest_roi.addScaleHandle([0, 1], [1, 0])  # Add a scale handle to allow resizing vertically

        self.video_view.addItem(self.interest_roi)
        self.roi_button2.setStyleSheet("background-color: green")

    def calculate(self):

        # Implement the logic to calculate the contrast to noise ratio
        if self.interest_roi is not None and self.background_roi is not None:
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