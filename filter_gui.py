"""
Shoreline Filters for CoastSeg (image suitability, segmentation filter, spatial KDE)
Mark Lundine, USGS
Requires tensorflow, pandas, numpy, pyqt, matplotlib, spatial-kde, rasterio, gdal, python=3.10
"""
#basic imports
import os
import glob
import sys
import shutil
#pyqt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
#filters
import image_segmentation_filter
import image_filter 
import shoreline_change_envelope

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        sizeObject = QDesktopWidget().screenGeometry(-1)
        global screenWidth
        screenWidth = sizeObject.width()
        global screenHeight
        screenHeight = sizeObject.height()
        global bw1
        bw1 = int(screenWidth/15)
        global bw2
        bw2 = int(screenWidth/50)
        global bh1
        bh1 = int(screenHeight/15)
        global bh2
        bh2 = int(screenHeight/20)

        self.setWindowTitle("CoastSeg Filters")
        self.home()
        
    def run_image_suitability_filter(self, threshold):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        home = str(QFileDialog.getExistingDirectory(self, "Select RGB Image Folder"))
        if home:        
            image_filter.run_inference_rgb(os.path.join(os.getcwd(), 'models', 'image_rgb', 'best.h5'),
                                           home,
                                           home,
                                           os.path.join(home, 'good_bad.csv'),
                                           threshold
                                           )
        
    def run_multi_image_suitability_filter(self, threshold):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        home = str(QFileDialog.getExistingDirectory(self, "Select Sessions Folder"))
        if home:        
            image_filter.inference_multiple_sessions(home, threshold, model='rgb')

    def run_segmentation_filter(self, threshold):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        home = str(QFileDialog.getExistingDirectory(self, "Select RGB Segmentation Image Folder"))
        if home:              
            image_segmentation_filter.run_inference_rgb(os.path.join(os.getcwd(), 'models', 'segmentation_rgb', 'best_seg.h5'),
                                                        os.path.join(home, 'jpg_files', 'preprocessed', 'RGB'),
                                                        os.path.join(home, 'jpg_files', 'preprocessed', 'RGB'),
                                                        os.path.join(home, 'good_bad.csv'),
                                                        threshold
                                                        )
        
    def run_multi_image_segmentation_filter(self, threshold):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        home = str(QFileDialog.getExistingDirectory(self, "Select Sessions Folder"))
        if home:        
            image_segmentation_filter.inference_multiple_sessions(home, threshold)

    def run_spatial_kde_filter(self, radius, cell_size, buffer):
        options = QFileDialog.Options()
        shorelines, _ = QFileDialog.getOpenFileName(self,"Select Extracted Shorelines Points GeoJSON", "","GeoJSON (*.geojson)", options=options)
        if shorelines:
            extracted_shorelines_points = shorelines
            site = os.path.dirname(shorelines)
            point_density_kde_path =  os.path.join(site, 'spatial_kde.tif')
            otsu_path = os.path.join(site, 'spatial_kde_otsu.tif')
            shoreline_change_envelope_path = os.path.join(site, 'shoreline_change_envelope.geojson')
            shoerline_change_envelope_buffer_path = os.path.join(site, 'shoreline_change_envelope_buffer.geojson')
            shoreline_change_envelope.get_point_density_kde(extracted_shorelines_points,
                                                            point_density_kde_path,
                                                            otsu_path,
                                                            shoreline_change_envelope_path,
                                                            shoreline_change_envelope_buffer_path,
                                                            kde_radius=buffer,
                                                            cell_size=cell_size
                                                            )
        
    def run_multi_spatial_kde_filter(self, radius, cell_size, buffer):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        home = str(QFileDialog.getExistingDirectory(self, "Select Sessions Folder"))
        if home:        
            get_point_density_kde_multiple_sessions(home,
                                                    kde_radius=80,
                                                    cell_size=15,
                                                    buffer=50)
        
    def home(self):
        self.scroll = QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()                 # Widget that contains the collection of Vertical Box
        self.vbox = QGridLayout()             # The Vertical Box that contains the Horizontal Boxes of  labels and buttons
        self.widget.setLayout(self.vbox)

        #image suitability
        image_suitability_filter = QPushButton('Image Suitability Filter Single Session')
        self.vbox.addWidget(image_suitability_filter , 0, 0)
        #image suitability
        image_suitability_filter_multi = QPushButton('Image Suitability Filter Multiple Sessions')
        self.vbox.addWidget(image_suitability_filter_multi , 1, 0)
        
        #image suitability threshold
        image_suitability_threshold_label = QLabel('Threshold')
        image_suitability_threshold = QDoubleSpinBox()
        image_suitability_threshold.setMinimum(0.01)
        image_suitability_threshold.setMaximum(0.99)
        image_suitability_threshold.setValue(0.437)
        self.vbox.addWidget(image_suitability_threshold_label, 2, 0)
        self.vbox.addWidget(image_suitability_threshold, 3, 0)

        #segmentation filter button
        segmentation_filter = QPushButton('Segmentation Filter Single Session')
        self.vbox.addWidget(segmentation_filter, 0, 1)
        
        #segmentation filter multiple button
        segmentation_filter_multi = QPushButton('Segmentation Filter Multiple Sessions')
        self.vbox.addWidget(segmentation_filter_multi, 1, 1)

        
        #segmentation filter threshold
        segmentation_filter_threshold_label = QLabel('Threshold')
        segmentation_filter_threshold = QDoubleSpinBox()
        segmentation_filter_threshold.setMinimum(0.01)
        segmentation_filter_threshold.setMaximum(0.99)
        segmentation_filter_threshold.setValue(0.482)
        self.vbox.addWidget(segmentation_filter_threshold_label, 2, 1)
        self.vbox.addWidget(segmentation_filter_threshold, 3, 1)

        #spatial kde button
        spatial_kde_filter = QPushButton('Spatial KDE Filter Single Session')
        self.vbox.addWidget(spatial_kde_filter, 0, 2)
        
        #spatial kde button
        spatial_kde_filter_multi = QPushButton('Spatial KDE Filter Multiple Sessions')
        self.vbox.addWidget(spatial_kde_filter_multi, 1, 2)
        
        #spatial kde radiuds
        radius_label = QLabel('Radius (m)')
        radius_slider = QSpinBox()
        radius_slider.setMinimum(1)
        radius_slider.setMaximum(100)
        radius_slider.setValue(80)
        self.vbox.addWidget(radius_label, 2, 2)
        self.vbox.addWidget(radius_slider, 3, 2)

        #cell size
        cell_size_label = QLabel('Cell Size (m)')
        cell_size_slider = QSpinBox()
        cell_size_slider.setMinimum(1)
        cell_size_slider.setMaximum(50)
        cell_size_slider.setValue(15)
        self.vbox.addWidget(cell_size_label, 4, 2)
        self.vbox.addWidget(cell_size_slider, 5, 2)

        #buffer
        buffer_label = QLabel('Buffer (m)')
        buffer_slider = QSpinBox()
        buffer_slider.setMinimum(0)
        buffer_slider.setMaximum(50)
        buffer_slider.setValue(15)
        self.vbox.addWidget(buffer_label, 6, 2)
        self.vbox.addWidget(buffer_slider, 7, 2)
        
        #Actions
        image_suitability_filter.clicked.connect(lambda: self.run_image_suitability_filter(image_suitability_threshold.value()))
        image_suitability_filter_multi.clicked.connect(lambda: self.run_multi_image_suitability_filter(image_suitability_threshold.value()))
        segmentation_filter.clicked.connect(lambda: self.run_image_segmentation_filter(segmentation_filter_threshold.value()))
        segmentation_filter_multi.clicked.connect(lambda: self.run_multi_image_segmentation_filter(segmentation_filter_threshold.value()))
        spatial_kde_filter.clicked.connect(lambda: self.run_spatial_kde_filter(radius_slider.value(), cell_size_slider.value(), buffer_slider.value()))
        spatial_kde_filter_multi.clicked.connect(lambda: self.run_multi_spatial_kde_filter())
        
        #Scroll policies
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)
        self.setCentralWidget(self.scroll)


## Function outside of the class to run the app   
def run():
    app = QApplication(sys.argv)
    GUI = Window()
    GUI.show()
    sys.exit(app.exec_())

## Calling run to run the app
run()
