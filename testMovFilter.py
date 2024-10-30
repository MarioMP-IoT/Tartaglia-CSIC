from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget
import cv2
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import os, copy
from scipy import signal
from numba import jit
import scipy.ndimage as ndi
from time import sleep
import cProfile, pstats
import multiprocessing as mp
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager
from file_functions import *



profiler = cProfile.Profile()
profiler.enable()

def update_result(N,queue,queue_processing, control_flag):
    #funcion para el proceso secundario que actualiza la imagen filtrada en N
    N=N
    sum =0
    while True:
        if not queue.empty() and control_flag.value:
            [frame]=queue.get()
            if sum==0:
                frames=np.expand_dims(frame, axis=2)
                sum+=1
            else:
                frames=np.append(frames, np.expand_dims(frame, axis=2),axis=2)
                sum += 1
            if control_flag.value and sum ==N:
                frame_result=np.mean(frames, axis=2)
                #Calculamos la envolvente resultante del filtro
                if queue_processing.empty():
                    queue_processing.put([frame_result])
                sum=sum-1
                frames=frames[:,:,1:]

        else:

            pass#donothing


class MainWindow(QtGui.QWidget):
    #frame_data = QtCore.pyqtSignal(np.ndarray) #señal para emitir el frame leido por el capturador para que sea guardado por el hilo de guardado

    #Por defecto incluimos estos parametros que podran ser editados desde la app
    N_LINES=82
    N_SAMPLES=866


    def __init__(self,N, queue_data,queue_processing, control_flag):
        super().__init__()

        self.setupUi()
        self.queue_data=queue_data
        self.queue_processing = queue_processing
        self.N=N
        #self.real_time=real_time
        self.control_flag=control_flag
        #self.mp_paint_flag=mp_paint_flag


        # Crear un temporizador para actualizar el widget a una tasa maxima posible
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0)
        # Crear un temporizador para actualizar el procesado
        self.timer_proc = QtCore.QTimer()
        self.timer_proc.timeout.connect(self.update_processing)
        self.timer_proc.start(0)




    def setupUi(self):
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setWindowTitle('Video Capturer')
        self.resize(1200, 800)
        #self.showFullScreen()
        # self.centralwidget = QtGui.QWidget(self)
        # self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtGui.QGridLayout()
        self.setLayout(self.gridLayout)
        # self.stack = QtGui.QStackedWidget(self.centralwidget)
        # self.stack.setObjectName("stack")
        self.frame_wg = QtGui.QWidget()
        self.frame_wg.setObjectName("frame_wg")

        self.HLayout = QtGui.QHBoxLayout()
        #--------TODO: ALTERNATIVA
        # self.main_graphicsView=pg.GraphicsView()
        #
        # self.main_vb = pg.ViewBox()
        # self.main_vb.invertY()
        # self.main_graphicsView.setCentralItem(self.main_vb)
        # self.main_vb.setAspectLocked()
        # self.main_img = pg.ImageItem()
        # self.main_vb.addItem(self.main_img)
        #_________________

        self.frame_wg.setLayout(self.HLayout)
        self.frame_wg.adjustSize()


        self.rawGLImg = CustomRawImageGLWidget(scaled=True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        #sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rawGLImg.sizePolicy().hasHeightForWidth())

        self.rawGLImg.setSizePolicy(sizePolicy)


        self.rawGLImg.setObjectName("rawGLImg")
        self.HLayout.addWidget(self.rawGLImg)

        #widget vision filtrada
        self.rawGLImg2 = CustomRawImageGLWidget(scaled=True)
        sizePolicy2 = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        # sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.rawGLImg2.sizePolicy().hasHeightForWidth())

        self.rawGLImg2.setSizePolicy(sizePolicy2)

        self.rawGLImg2.setObjectName("rawGLImg2")
        self.HLayout.addWidget(self.rawGLImg2)

        self.gridLayout.addWidget(self.frame_wg, 1, 0, 1, 1)
        #self.gridLayout.addWidget(self.main_graphicsView, 1, 0, 1, 1)
        # self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 1)
        self.fps_label = QtGui.QLabel()
        self.fps_label.setObjectName("FPS")
        self.fps_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fps_label.setStyleSheet('color: black; font-size: 10pt; font-weight: bold')
        self.fps_label.setFixedSize(150, 25)
        self.fps_label.move(self.width() - self.fps_label.width() - 20, 20)

        #self.gridLayout.addWidget(self.fps_label, 0, 0, 1, 1)







        #self.gridLayout.addWidget(self.cb, 0, 1, 1, 1)

        #Configuro el tamaño del widget
        self.rawGLImg.setMinimumSize(960, 540)
        #self.rawGLImg.setMaximumSize(1920, 1080)

        #configuro componente para grabar video
        self.elements_wg = QtGui.QWidget()
        self.elements_wg.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.elements_wg.setContentsMargins(0, 0, 0, 0)
        self.start_stop_button = QtGui.QPushButton("Save")
        self.processing_button = QtGui.QPushButton("Start processing")
        self.seconds_spin_box = QtGui.QSpinBox()
        self.seconds_spin_box.setFixedSize(70, 25)
        self.filename_line_edit = QtGui.QLineEdit()

        self.load_video_button = QtGui.QPushButton('Load Video', self)


        h_layout = QtGui.QHBoxLayout()
        h_layout.addWidget(self.fps_label)

        h_layout.addWidget(QtGui.QLabel("Seconds to record:"))
        h_layout.addWidget(self.seconds_spin_box)
        h_layout.addWidget(QtGui.QLabel("Filename:"))
        h_layout.addWidget(self.filename_line_edit)
        h_layout.addWidget(self.load_video_button)
        h_layout.addWidget(self.start_stop_button)
        h_layout.addWidget(self.processing_button)



        self.elements_wg.setLayout(h_layout)
        self.elements_wg.adjustSize()
        self.gridLayout.addWidget(self.elements_wg, 0, 0, 1, 2)



        # _________________
        # self.processing_frame_wg = QtGui.QWidget()
        # self.processing_frame_wg.setObjectName("frame_wg")
        # self.HLayout_2 = QtGui.QHBoxLayout()
        #
        # self.processing_frame_wg.setLayout(self.HLayout_2)
        #
        # self.rawGLImg_proc = RawImageGLWidget(scaled=True)
        # # sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        # # # sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        # # sizePolicy.setHorizontalStretch(0)
        # # sizePolicy.setVerticalStretch(0)
        # # sizePolicy.setHeightForWidth(self.rawGLImg_proc .sizePolicy().hasHeightForWidth())
        # #
        # # self.rawGLImg_proc .setSizePolicy(sizePolicy)
        # self.rawGLImg_proc .setObjectName("rawGLImg_proc")
        # self.HLayout_2.addWidget(self.rawGLImg_proc)
        # self.gridLayout.addWidget(self.processing_frame_wg, 1, 2, 1, 1)




        # Inicializar las variables de fps

        #self.fps = 0
        self.lastTime = ptime.time()
        self.fps = None
        self.frames_recorded = 0
        self.video_pos = 0
        self.recording=False
        self.frame = None
        self.writer = None
        self.loaded=False

        # Inicializar la lista de esquinas
        self.corners = []
        #Inicializo objeto con info
        #self.info2Proc=InfoToProcess()


        #Ruta donde guardar adquisiciones
        self.path_to_save = '.\\video_capture\\'
        CHECK_FOLDER = os.path.isdir(self.path_to_save)
        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.makedirs(self.path_to_save)

        self.start_stop_button.clicked.connect(self.toggle_recording)
        self.processing_button.clicked.connect(self.toggle_processing)

        self.load_video_button.clicked.connect(self.load_video)


        # Conectar la señal de clic del mouse a la función
        # self.main_graphicsView.scene().sigMouseClicked.connect(self.mouse_click_event)
        # self.main_graphicsView.useOpenGL()
        # self.graphicsView2.useOpenGL()

        self.rawGLImg.clicked.connect(self.mouse_click)



        #self.video_recorder = None

    def toggle_recording(self):
        if  not self.recording:

            self.start_recording()
        else:
            self.stop_recording()
    def toggle_processing(self):
        if  control_flag.value:

            control_flag.value = 0
            self.processing_button.setText("Start processing")
            self.t1.join()

        else:
            control_flag.value = 1
            self.processing_button.setText("Stop processing")
            self.t1 = Process(target=update_result, args=(self.N, self.queue_data, self.queue_processing, control_flag,))
            self.t1.start()






    def load_video(self):
        self.video_file, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open Video', '',
                                                               'Video Files (*.BIN) ;; PNG files (*.png)')
        self.extension = self.video_file.split('.')[-1]
        if self.video_file and (self.extension == 'BIN' or self.extension == 'bin'):
            # Leo el archivo y genero un objeto 'BinFile' que contiene los datos crudos tal cual se leen del archivo
            bfile = BinFile(self.video_file)
            # A partir del objeto 'BinFile' genero un objeto 'Dataset' que ya tiene los datos en formato 'cubo de datos'
            # y contiene las matrices de coordenadas de los píxeles para poder pintar
            self.dset = Dataset(bfile, resample=0, med_kernel=0, avg_kernel=0)

            for i in range(self.dset.nimg):

                #aplicamos filtro paso banda

                # b = signal.firwin(63, [1e6, 5e6], pass_zero='bandpass', fs=40e6, scale=True)
                # a = 1
                # # Filtrar la matriz cruda B-scan con el filtro paso banda
                # filtered_b_scan = signal.filtfilt(b, a, self.dset.bscan[:, :, i], axis=0)

                envolvente = signal.hilbert(self.dset.bscan[:, :, i], axis=0)
                envolvente = 20 * np.log10(np.abs(envolvente) / 3276.7)  # TAMAÑO INT16 pasamos a dB
                envolvente[envolvente < -self.dset.dB_range] = -self.dset.dB_range
                self.dset.bscan[:, :, i] = envolvente

            self.dset.ScanConvert(raw=False)  # como bscan y raw son iguales trabajo con uno para filtrar y el otro pinto para comparar






            self.loaded = True

            # self.update()
            # self.scroll.show()
            del bfile


    def mouse_click_event(self, ev):
        if ev.button() == 1:
            # Obtener la posición del clic en coordenadas de la escena
            pos = self.main_vb.mapSceneToView(ev.scenePos())

            # Agregar la posición a la lista de esquinas
            self.corners.append([pos.x(), pos.y()])

            # Dibujar un punto rojo en la posición del clic
            # self.main_vb.plot([pos.x()], [pos.y()], pen=None, symbol='o', symbolSize=10, symbolBrush='r')

            # Si se han marcado las cuatro esquinas, imprimir la lista de coordenadas
            if len(self.corners) == 4:
                self.corners=np.array(self.corners,dtype=np.int32)
                # conectar función de selección de esquinas
                self.select_corners()
    def mouse_click(self, x, y):

            # Obtener la posición del clic en coordenadas de la escena


            # Agregar la posición a la lista de esquinas
            if len(self.corners) < 4:
                self.corners.append([x, y])

            # Dibujar un punto rojo en la posición del clic
            # self.main_vb.plot([pos.x()], [pos.y()], pen=None, symbol='o', symbolSize=10, symbolBrush='r')

            # Si se han marcado las cuatro esquinas, imprimir la lista de coordenadas
            if len(self.corners) == 4:
                self.corners=np.array(self.corners,dtype=np.int32)
                # conectar función de selección de esquinas
                self.select_corners()
    # def select_corners(self):
    #     #roi_new = pg.PolyLineROI(self.corners)
    #     #primero debemor reordenar los puntos
    #     #unavez tenemos los 4 puntos marcados comprobamos si se trata de una imagen sectorial o rectangular
    #     adjusted = self.adjust_corners()
    #     if adjusted:
    #         self.maskForm=self.check_form()
    #         if self.maskForm=='square':
    #
    #
    #             pts=self.corners
    #             #pts=self.get_pts_rectangle(self.corners)
    #             # Obtén las coordenadas x e y de las esquinas
    #             x_coords = [c[0] for c in pts]
    #             y_coords = [c[1] for c in pts]
    #
    #             # Calcula la posición y el tamaño de la ROI
    #             x = min(x_coords)
    #             y = min(y_coords)
    #             w = max(x_coords) - x
    #             h = max(y_coords) - y
    #
    #             # Creo la ROI con los valores calculados
    #             # self.pts_roi=[(x, y), (w, h)]
    #             # self.roi = pg.RectROI([x, y], [w, h],movable=False,rotatable=False, resizable=False,removable=True, aspectLocked=True)
    #             #self.bscanParams=BscanPatameters(x=x,y=y,w=w,h=h) #Sustituyo la clase por un diccionario
    #             self.bscanParams={'form':self.maskForm,'x':x,'y':y,'w':w,'h':h, 'nlin':self.N_LINES, 'nr':self.N_SAMPLES}#engloba los parametros necesarios para formar el bscan a partir del frame
    #             self.paint_flag = True  # activo bandera para que pinte la region seleccionada
    #             self.regionMarked = True  # mientras se pregunta pintamos la region en pantalla
    #
    #             #self.main_vb.addItem(self.roi)
    #             #self.lw.addItem(self.roi)
    #             #self.roi.sigRegionChanged.connect(self.update_roi_mask)
    #             # Una vez calculamos los parametros preguntamos para que confirme si es la imagen que deseaba
    #
    #             self.saveROI.emit()
    #
    #
    #         elif self.maskForm=='sectorial':
    #             self.sectorial_mask = np.zeros((self.frame.shape[0], self.frame.shape[1]), dtype=np.uint8)
    #             #-----GETPTS
    #
    #
    #             center, r1, r2, angle = calculate_center_radii(self.corners)
    #             #self.bscanParams = BscanPatameters(center=center, r1=r1, r2=r2, angle=angle)#Sustituyo esta linia añadiendo parametros al diccionario
    #             self.bscanParams = {'form':self.maskForm,'center': center, 'r1': r1, 'r2': r2, 'angle': angle,'nlin':self.N_LINES, 'nr':self.N_SAMPLES}#engloba los parametros necesarios para formar el bscan a partir del frame
    #
    #             #TODO: QUizas aqui estoy haciendo cosas repetidas o que se pudieran optimizar en la creacion del bscan
    #             pts = calculate_ROI(center, r1, r2, angle)
    #
    #             #pts=self.get_pts(self.corners)
    #             self.pts_roi = np.array(pts, dtype=np.int32)
    #             cv2.fillPoly(self.sectorial_mask, [self.pts_roi], True)
    #             # Obtener el rectángulo delimitador
    #             x, y, w, h = cv2.boundingRect(self.sectorial_mask)
    #
    #             self.dimensions_roi=[x, y, w, h]
    #             new_center = [center[0] - x, center[1] - y]
    #             cx=new_center[0]
    #             cy = new_center[1]
    #             # Generate evenly spaced angles
    #             angles = np.linspace(-angle / 2, angle / 2, self.N_LINES)
    #
    #
    #             self.frame_nx = w
    #             self.frame_ny = h
    #             ri = r1
    #             rf = r2
    #             dr =(rf-ri)/self.N_SAMPLES #salto entre muestras
    #             da = (angles[1] - angles[0]) #Salto entre angulos
    #
    #             # Calculo las matrices de índices y pesos del interpolador
    #             self.ix = np.zeros((self.frame_ny, self.frame_nx, 2), dtype=int)
    #             self.w_p = np.zeros((self.frame_ny, self.frame_nx, 4), dtype=float)
    #             self.frames_val = np.zeros((self.frame_ny, self.frame_nx), dtype=bool)
    #             ScanConverterIndexes_Bilineal(self.frame_ny, self.frame_nx, cx, cy, ri, rf, da, dr, angles, self.ix,
    #                                           self.frames_val,
    #                                           self.N_LINES, self.N_SAMPLES, self.w_p)
    #
    #             #cv2.drawContours(self.sectorial_mask, pts, 0, color=(255, 255, 255), thickness=cv2.FILLED)
    #
    #             #self.roi= pg.PolyLineROI(pts,closed=True, movable=False,rotatable=False, resizable=False,removable=True, aspectLocked=True)
    #             #self.main_vb.addItem(self.roi)
    #             #self.lw.addItem(self.roi)
    #             #self.roi.sigRegionChanged.connect(self.update_roi_mask)
    #             self.paint_flag = True  # activo bandera para que pinte la region seleccionada
    #             self.regionMarked = True  # mientras se pregunta pintamos la region en pantalla
    #             # Una vez calculamos los parametros preguntamos para que confirme si es la imagen que deseaba
    #
    #             self.saveROI.emit()
    #
    #         else:
    #             self.errorROI.emit()
    #
    #


    # def check_form(self, threshold=7):
    #     # Calcular los vectores que forman los lados adyacentes del cuadrilátero
    #     vector1 = self.corners[1] - self.corners[0]
    #     vector2 = self.corners[2] - self.corners[1]
    #     vector3 = self.corners[3] - self.corners[2]
    #     vector4 = self.corners[0] - self.corners[3]
    #
    #     # Calcular los ángulos entre los vectores adyacentes
    #     angle1 = np.degrees(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))
    #     angle2 = np.degrees(np.arccos(np.dot(vector2, vector3) / (np.linalg.norm(vector2) * np.linalg.norm(vector3))))
    #     angle3 = np.degrees(np.arccos(np.dot(vector3, vector4) / (np.linalg.norm(vector3) * np.linalg.norm(vector4))))
    #     angle4 = np.degrees(np.arccos(np.dot(vector4, vector1) / (np.linalg.norm(vector4) * np.linalg.norm(vector1))))
    #
    #     # Verificar si los ángulos están dentro del umbral dado para ser considerados cercanos a 90 grados
    #     if abs(angle1 - 90) < threshold and abs(angle2 - 90) < threshold and abs(angle3 - 90) < threshold and abs(
    #             angle4 - 90) < threshold:
    #         form='square'
    #         return form
    #     elif are_similar(angle1,angle4) and are_similar(angle2,angle3):
    #         form = 'sectorial'
    #         return form
    #     else:
    #         form='unknown'
    #         return form


    # def getBscan(self, params, img):
    #     if self.maskForm=='square':
    #         x = params['x']
    #         y = params['y']
    #         w = params['w']
    #         h = params['h']
    #         bscan=cv2.resize(img,(self.N_LINES,self.N_SAMPLES),interpolation=cv2.INTER_AREA)
    #         return bscan
    #     elif self.maskForm=='sectorial':
    #         center=params['center']
    #         r1=params['r1']
    #         r2=params['r2']
    #         angle=params['angle']
    #         bscan = create_bscan(data=img, center=center, theta=angle, r1=r1, r2=r2, n_lines=self.N_LINES, n_samples=self.N_SAMPLES)
    #         #bscan=ndi.uniform_filter(bscan, size=7).astype(np.uint8)
    #         return bscan




    def video_recorder(self, frame):


        if self.filename_toRecord == "" or self.seconds_toRecord == 0:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Debe insertar nombre y segundos para poder guardar adquisición")
            msg.setWindowTitle("Información")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            self.stop_recording()
        else:
            # Obtiene el tiempo actual
            t_actual = cv2.getTickCount()
            # Calcula el tiempo transcurrido en segundos
            t_transcurrido = (t_actual - self.t_inicio) / cv2.getTickFrequency()
            if t_transcurrido<self.seconds_toRecord: #self.frames_recorded < self.total_frames_toRecord:

                if frame is not None:

                    # Escribe el frame al archivo
                    self.writer.write(frame)
                    self.frames_recorded += 1
            else:
                self.stop_recording()


    def start_recording(self):
        self.filename_toRecord = self.filename_line_edit.text()
        self.seconds_toRecord = self.seconds_spin_box.value()
        self.fps_toRecord = int(self.fps)#self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames_toRecord = int(self.seconds_toRecord * self.fps_toRecord)
        self.frames_recorded = 0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path_to_save + self.filename_toRecord + '.mp4', fourcc, self.fps_toRecord, (self.frame_WIDTH, self.frame_HEIGHT))
        # Obtiene el tiempo de inicio de la grabación
        self.t_inicio = cv2.getTickCount()

        self.recording = True
        self.start_stop_button.setText("Stop")


    def stop_recording(self):
        #reseteo variables
        # Liberamos los recursos
        if self.writer is not None:
            self.writer.release()
        # reseteo variables
        self.writer=None
        self.recording = False

        self.start_stop_button.setText("Start")
    # def update_roi_mask(self):
    #     #if self.video_frame is not None and self.roi is not None:
    #     if self.roi is not None:
    #         #img_roi = self.roi.getArrayRegion(self.frame, self.main_img).astype('uint8')
    #         x1, y1 = self.corners[0]
    #         x2, y2 = self.corners[2]
    #         img_roi=self.frame[int(y1):int(y2), int(x1):int(x2)].astype('uint8')
    #         img_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY)
    #
    #         # Actualizar el widget de imagen a procesar
    #         self.img2.setImage(img_roi,autolevels=False)

    def update(self): #Funcion para actualizar la imagen sin filtro

        try:

            if self.loaded and self.video_pos<self.dset.nimg:
                self.frame= self.dset.frames[:,:,self.video_pos]

                # Actualizar el widget de imagen cruda
                self.rawGLImg.setImage(self.frame, levels=[-self.dset.dB_range,0])
                if control_flag.value:
                    self.queue_data.put([self.frame])
                self.video_pos += 1
            elif self.loaded and self.video_pos>=self.dset.nimg:

                #reiniciamos la posicion para printar el bucle
                self.video_pos = 0
                self.frame = self.dset.frames[:, :, self.video_pos]
                # Actualizar el widget de imagen cruda
                self.rawGLImg.setImage(self.frame, levels=[-self.dset.dB_range,0])
                if control_flag.value:
                    self.queue_data.put([self.frame])
                self.video_pos += 1

            # Actualizar la etiqueta de fps
            now = ptime.time()
            dt = now - self.lastTime
            self.lastTime = now
            if self.fps is None:
                self.fps = 1.0 / dt
            else:
                s = np.clip(dt * 3., 0, 1)
                self.fps = self.fps * (1 - s) + (1.0 / dt) * s

            self.fps_label.setText('%0.2f fps' % self.fps)
        except Exception as e:
            print(e)

    def update_processing(self):


        # compruebo si la cola esta vacia y hago put para procesar
        if not self.queue_processing.empty() and control_flag.value:# and mp_paint_flag.value:


            [frame_result]=self.queue_processing.get()

            self.rawGLImg2.setImage(frame_result, levels=[-self.dset.dB_range,0])
            if self.recording:
                self.video_recorder(frame_result)

                #mp_paint_flag.value=0






    # def get_pts(self, corners_prev):
    #
    #     if corners_prev is None:
    #         return
    #     adjusted = self.adjust_corners()
    #     if adjusted:
    #         center, r1, r2, angle = calculate_center_radii(self.corners)
    #         points = calculate_ROI(center, r1, r2, angle)
    #
    #         return points
    #
    # def get_pts_rectangle(self, corners_prev):
    #
    #     if corners_prev is None:
    #         return
    #     corners = self.adjust_corners()
    #     # center, r1, r2, angle = calculate_center_radii(corners)
    #     # points = calculate_ROI(center, r1, r2, angle)
    #
    #     return corners



    def closeEvent(self, event):

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
        event.accept()

# #@jit(nopython=True, parallel=False, cache=True)
# def maskConvert(mask, dimensions_roi, bscanParams, frame):
#     center=[bscanParams['center'][0]-dimensions_roi[2],bscanParams['center'][1]-dimensions_roi[3]]
#     print(center)
#     frame_nx=dimensions_roi[2]video_pos
#     frame_ny = dimensions_roi[3]
#     r1=bscanParams['r1']
#     r2 = bscanParams['r2']
@jit(nopython=True, parallel=False, cache=True)
def ScanConvert_Bilineal(frame_ny, frame_nx, val, frame, bscan, ix, w):

    for i in range(frame_ny):
        for j in range(frame_nx):
            if val[i, j]:

                # Indices a la matriz
                ixr1  = ix[i,j,0]
                ixr2  = ixr1 + 1
                ixth1 = ix[i,j,1]
                ixth2 = ixth1 + 1

                frame[i, j] = w[i, j, 0] * bscan[ixr1, ixth1] + \
                                         w[i, j, 1] * bscan[ixr2, ixth1] + \
                                         w[i, j, 2] * bscan[ixr1, ixth2] + \
                                         w[i, j, 3] * bscan[ixr2, ixth2]
    return frame

@jit(nopython=True, parallel=False, cache=True)
def ScanConverterIndexes_Bilineal(frame_ny, frame_nx, cx, cy, ri, rf, da, dr, angles, ix, val, nlin, nr, w):

    # matrices de índices a las muestras y pixel válido
    for i in range(frame_ny):
        for j in range(frame_nx):
            d = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)  # Distancia al centro
            alfa = np.arcsin((j - cx) / d)  # Ángulo con respecto a la vertical

            if (d > ri) and (d < rf) and (alfa > angles[0]) and (alfa < angles[-1]):
                ixr = (d - ri) / dr
                ixth = (alfa - angles[0]) / da

                ix[i, j, 1] = np.floor(ixth)  # ángulo
                ix[i, j, 0] = np.floor(ixr)  # radio

                if (ix[i, j, 1] >= 0) and (ix[i, j, 1] < nlin - 1) and (ix[i, j, 0] >= 0) and (ix[i, j, 0] < nr - 1):

                    # Calculo las proporciones
                    dn = ixr - np.floor(ixr)
                    dn1 = np.ceil(ixr) - ixr
                    thm = ixth - np.floor(ixth)
                    thm1 = np.ceil(ixth) - ixth

                    # Calculo los pesos
                    w[i,j,0] = dn1 * thm1
                    w[i,j,1] = dn  * thm1
                    w[i,j,2] = dn1 * thm
                    w[i,j,3] = dn  * thm

                    val[i, j] = True





class CustomRawImageGLWidget(RawImageGLWidget):
    clicked = QtCore.pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            # pos=self.mapToParent(event.pos())
            x_conv_factor=self.opts[0].shape[1]/self.size().width()
            y_conv_factor = self.opts[0].shape[0]/self.size().height()
            x = int(event.pos().x()*x_conv_factor)
            y = int(event.pos().y()*y_conv_factor)
            self.clicked.emit(x, y)
        super().mousePressEvent(event)

    def setRawImage(self, image, autoLevels=True, levels=None):
        # get the dimensions of the image and the widget
        img_height, img_width = image.shape[:2]
        widget_height, widget_width = self.height(), self.width()

        # calculate the scale factor to maintain the aspect ratio of the image
        width_scale = widget_width / img_width
        height_scale = widget_height / img_height
        scale_factor = min(width_scale, height_scale)

        # calculate the dimensions of the scaled image
        scaled_width = int(img_width * scale_factor)
        scaled_height = int(img_height * scale_factor)

        # scale the image using cv2
        scaled_image = cv2.resize(image, (scaled_width, scaled_height),
                                  interpolation=cv2.INTER_AREA)

        # set the scaled image in the widget
        super().setScaledImage(scaled_image, autoLevels, levels)
        self.setFixedSize(scaled_width, scaled_height)

# class BscanPatameters:
#     '''Clase que engloba los parametros necesarios para formar el bscan a partir del frame'''
#
#     def __init__(self, x=None, y=None, w=None, h=None, center=None, angle=None, r1=None, r2=None):
#         #Parametros para imagenes rectangulares
#         self.x=x
#         self.y=y
#         self.w=w
#         self.h=h
#         #Parametros para imagenes angulares
#         self.center=center
#         self.angle=angle
#         self.r1=r1
#         self.r2=r2

class InfoToProcess: #TODO: ES MAS EFICIENTE EN VELOCIDAD SI TRABAJO CON DICT
    '''Clase que engloba la informacion necesaria para poder ser procesada en el hilo de procesado'''
    def __init__(self, frame=None, bscan=None):
        self.frame=frame
        self.bscan=bscan

    def setData(self, frame=None, bscan=None):
        self.frame = frame
        self.bscan = bscan

if __name__ == '__main__':
    app = QtGui.QApplication([])
    N=5 #numero de imagenes con los que hacer el filtro
    q_data = Queue(maxsize=1)#cola de tamaño 1
    q_processed = Queue(maxsize=1)
    control_flag = mp.Value('i', 0)
    # mp_paint_flag = mp.Value('i', 0)


    # # # declaro objeto de tiempo real y empiezo hilo secundario
    # BaseManager.register('real_timeInfo', real_timeInfo)
    # manager = BaseManager()
    # manager.start()
    # real_time = manager.real_timeInfo()
    # win=MainWindow(real_time,control_flag)
    win = MainWindow(N,q_data,q_processed, control_flag)
    win.show()
    # Crear un temporizador para actualizar el procesado
    # timer_proc = QtCore.QTimer()
    # timer_proc.timeout.connect(win.update_processing)
    # timer_proc.start(0)
    # # #
    # t1 = Process(target=update_prediction, args=(real_time,control_flag,))
    # t1 = Process(target=update_prediction, args=(q_data,q_processed, control_flag,))
    # t1.start()
    app.exec_()
