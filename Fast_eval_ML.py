import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QComboBox, QLineEdit, QFileDialog, QHBoxLayout, QButtonGroup, QRadioButton, QLabel, QCheckBox
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import random

######################################################################################################
# Import modules for Machine Learning

import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import sklearn.metrics as sm
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap

from sklearn import datasets


##########################################################################################################

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 200  # Left Position of the window relative to the screen
        self.top = 200  # Top Position of the window relative to the screen
        self.title = 'Fast evaluation of machine learning models performance'
        self.width = 924   # Window width size
        self.height = 600  # Window height size
        self.std_toogle_state = True
        self.custom_set = False
        self.initUI()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.plot_default()  # By default, plot nothing
        self.m.move(50, 50)  # Controls the canvas position relative to the parent


        ############# BUTTONS ######################
        button1 = QPushButton(self)
        # button.setToolTip('This s an example button')
        icon = QIcon("test.png")
        button1.setIcon(icon)
        button1.move(505, 0)
        button1.resize(50, 50)
        button1.clicked.connect(self.btn1_listener)

        self.button_set = QPushButton('Set Columns', self)
        self.button_set.move(650, 150)
        self.button_set.clicked.connect(self.btn_set_listener)
        self.button_set.setEnabled(False)

        self.button_batch = QPushButton('Set batch', self)
        self.button_batch.move(770, 10)
        self.button_batch.clicked.connect(self.btn_batch)
        self.button_batch.setEnabled(False)


        ############# LIST OF ELEMENTS ######################
        items = ['Logistic Regression', 'Support Vector Machine', 'Random Forest', 'K-nearest Neighbors', 'Gaussian Naive Bayes',
                 'Neural Network']

        self.lv = QComboBox(self)  # Self definition is only needed when a method for that specific object is used... outside the function containing it...
        self.lv.addItems(items)
        self.lv.move(130, 550)
        self.lv.resize(200, 20)
        self.lv.currentTextChanged.connect(self.itemChanged)
        self.lv.setEnabled(False)

        ############# TextEdit ######################
        self.edit = QLineEdit(self)
        self.edit.move(50, 15)
        self.edit.resize(450, 20)

        self.edit_start = QLineEdit(self)
        self.edit_start.move(640, 75)
        self.edit_start.resize(40, 20)

        self.edit_end = QLineEdit(self)
        self.edit_end.move(730, 75)
        self.edit_end.resize(40, 20)

        self.edit_label = QLineEdit(self)
        self.edit_label.move(700, 120)
        self.edit_label.resize(40, 20)



        self.edit_batch = QLineEdit(self)
        self.edit_batch.move(725, 15)
        self.edit_batch.resize(40, 20)




        ############ RADIO BUTTONS ################
        self.b1 = QRadioButton("Linear", self)
        self.b1.setChecked(True)
        self.b1.setEnabled(False)
        self.b1.toggled.connect(lambda: self.btnstate(self.b1))

        self.b2 = QRadioButton("Polynomial", self)
        self.b2.setEnabled(False)
        self.b2.toggled.connect(lambda: self.btnstate(self.b2))

        self.b3 = QRadioButton("RBF", self)
        self.b3.setEnabled(False)
        self.b3.toggled.connect(lambda: self.btnstate(self.b3))

        self.b4 = QRadioButton("Sigmoid", self)
        self.b4.setEnabled(False)
        self.b4.toggled.connect(lambda: self.btnstate(self.b4))

        self.b5 = QRadioButton("Cosine", self)
        self.b5.setEnabled(False)
        self.b5.toggled.connect(lambda: self.btnstate(self.b5))

        self.b1.move(30, 500)
        self.b2.move(130, 500)
        self.b3.move(250, 500)
        self.b4.move(350, 500)
        self.b5.move(450, 500)


        ######### LABELS ######
        l1 = QLabel(self)
        l1.setText("PCA Kernel")
        l1.resize(150, 20)
        l1.move(250, 480)

        l2 = QLabel(self)
        l2.setText("Classsifier")
        l2.resize(150, 20)
        l2.move(50, 548)

        l_feature = QLabel(self)
        l_feature.setText("Sample Features")
        l_feature.resize(150, 20)
        l_feature.move(635, 50)

        l_start = QLabel(self)
        l_start.setText("Start:")
        l_start.resize(35, 20)
        l_start.move(600, 75)

        l_end = QLabel(self)
        l_end.setText("End:")
        l_end.resize(30, 20)
        l_end.move(700, 75)

        l_end = QLabel(self)
        l_end.setText("Sample Labels:")
        l_end.resize(100, 20)
        l_end.move(600, 120)


        self.l_accuracy = QLabel(self)
        self.l_accuracy.setText("Model Accuracy:" )
        self.l_accuracy.resize(200, 15)
        self.l_accuracy.move(600, 350)

        self.l_precision = QLabel(self)
        self.l_precision.setText("Model Precision:")
        self.l_precision.resize(200, 15)
        self.l_precision.move(600, 370)

        self.l_recall = QLabel(self)
        self.l_recall.setText("Model Recall:")
        self.l_recall.resize(200, 15)
        self.l_recall.move(600, 390)

        self.l_f1 = QLabel(self)
        self.l_f1.setText("Model F1 score:")
        self.l_f1.resize(200, 15)
        self.l_f1.move(600, 410)



        ######## CHECK BOX ###########
        self.check_box = QCheckBox('Data Standardization', self)
        self.check_box.resize(200, 20)
        self.check_box.move(600, 200)
        self.check_box.stateChanged.connect(self.clickBox)

        self.check_box_train = QCheckBox('Custom training \n batch size (%)', self)
        self.check_box_train.resize(120, 40)
        self.check_box_train.move(600, 5)
        self.check_box_train.stateChanged.connect(self.clickBox_train)



        self.show()


    def btn1_listener(self):
        global start
        start = True
        global path_global
        print("Button 1 clicked")
        path_global = str(QFileDialog.getOpenFileName()[0])
        self.edit.setText(path_global)



        self.button_set.setEnabled(True)




    def btn_set_listener(self):
        print("Button SET clicked")
        global col_start, col_end, col_label, start, std_data
        start = True


        col_start = int(self.edit_start.text())
        col_end = int(self.edit_end.text())
        col_label = int(self.edit_label.text())
        print(col_start, col_end, col_label)


        if self.custom_set:
            X_train, y_train, cmap, markers, = load_dataset(path_global, train_size=self.train_size/100)

        else:
            X_train, y_train, cmap, markers, = load_dataset(path_global)

        self.X_train = X_train
        self.y_train = y_train
        self.cmap = cmap
        self.markers = markers
        self.kernel_opc = 1  # Default parameter
        self.opc = 1  # Default parameter
        std_data = self.check_box.isChecked()
        print(std_data)


        self.m.plot(self.opc, self.kernel_opc, self.X_train, self.y_train, self.cmap, self.markers)  # opc = 1 by default
        self.lv.setEnabled(True)
        self.b1.setEnabled(True)
        self.b2.setEnabled(True)
        self.b3.setEnabled(True)
        self.b4.setEnabled(True)
        self.b5.setEnabled(True)
        start = False
        self.b1.setChecked(True)
        self.lv.setCurrentIndex(0)  # Resets the classifier to Logistic regression (element index 0)


    def btn_batch(self):
        self.train_size = float(self.edit_batch.text())
        self.custom_set = True
        print(self.train_size)





    def itemChanged(self):
        self.opc = self.lv.currentIndex() + 1
        print("Classifier selected: ", self.opc)
        self.m.plot(self.opc, self.kernel_opc, self.X_train, self.y_train, self.cmap, self.markers)


    def btnstate(self, b):
        if b.text() == "Linear":
            if b.isChecked() == True:
                print('B1 is selected...')
                self.kernel_opc = 1
                self.m.plot(self.opc, self.kernel_opc, self.X_train, self.y_train, self.cmap,
                            self.markers)  # opc = 1 by default


        elif b.text() == "Polynomial":
            if b.isChecked():
                print('B2 is selected...')
                self.kernel_opc = 2
                self.m.plot(self.opc, self.kernel_opc, self.X_train, self.y_train, self.cmap,
                            self.markers)  # opc = 1 by default

        elif b.text() == "RBF":
            if b.isChecked():
                print('B3 is selected...')
                self.kernel_opc = 3
                self.m.plot(self.opc, self.kernel_opc, self.X_train, self.y_train, self.cmap,
                            self.markers)  # opc = 1 by default

        elif b.text() == "Sigmoid":
            if b.isChecked():
                print('B4 is selected...')
                self.kernel_opc = 4
                self.m.plot(self.opc, self.kernel_opc, self.X_train, self.y_train, self.cmap,
                            self.markers)  # opc = 1 by default

        elif b.text() == "Cosine":
            if b.isChecked():
                print('B5 is selected...')
                self.kernel_opc = 5
                self.m.plot(self.opc, self.kernel_opc, self.X_train, self.y_train, self.cmap,
                            self.markers)  # opc = 1 by default



    def clickBox(self, state):
        global std_data
        if state == QtCore.Qt.Checked:
            print('Checked')
            std_data = True
            self.std_toogle_state = True
            if not start:
                self.m.plot(self.opc, self.kernel_opc, self.X_train, self.y_train, self.cmap, self.markers)
            # print(std_data)

        else:
            print('Unchecked')
            std_data = False
            self.std_toogle_state = True

            if not start:
                self.m.plot(self.opc, self.kernel_opc, self.X_train, self.y_train, self.cmap, self.markers)
            # print(std_data)


    def clickBox_train(self, state):

        if state == QtCore.Qt.Checked:
            print('CB train Checked')
            self.button_batch.setEnabled(True)


        else:
            print('CB train Unchecked')
            self.button_batch.setEnabled(False)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # self.plot()
        self.previous_kernel = 0  # This assignation is needed because of a posterior 'if' decision in function plot

    def plot_default(self):
        ax = self.figure.add_subplot(111)
        # ax.plot(data, 'r-')
        ax.set_title('Graphical Result')
        self.draw()

    def plot(self, opc, kernel, X_train, y_train, cmap, markers):
        # global std_toogle_state
        self.cmap = cmap
        self.markers = markers


        # print("status toggle: ", ex.std_toogle_state)
        if std_data == True:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test_std = sc.transform(X_test)
            print('Data has been standardized')
            # X_test_std = sc.transform(X_test)


        ######### CLASSIFIERS ##########
        if opc == 1:  # SVM
            model = LogisticRegression()

        elif opc == 2:  # Logistic Regression
            model = SVC()

        elif opc == 3:  # Random forest
            model = RandomForestClassifier()

        elif opc == 4:  # K-nearest Neighbors
            model = KNeighborsClassifier()

        elif opc == 5:  # Gaussian Naive Bayes
            model = GaussianNB()

        elif opc == 6:  # Neural Network (Multi-Layer Perceptron)
            model = MLPClassifier()


        ##### KERNELS  #######
        # Do something to prevent having to calculate the Kernels every time a different classifier is selected...
        # if the same kernel is selected, do not re-calculate it again
        if kernel != self.previous_kernel or start or ex.std_toogle_state == True:
            if kernel == 1:  # Linear
                self.kernel_pca = KernelPCA(n_components=2, kernel='linear')
                self.X_train_pca = self.kernel_pca.fit_transform(X_train)
                self.previous_kernel = kernel
                ex.std_toogle_state = False

            elif kernel == 2:  # Polynomial
                self.kernel_pca = KernelPCA(n_components=2, kernel='poly')
                self.X_train_pca = self.kernel_pca.fit_transform(X_train)
                self.previous_kernel = kernel
                ex.std_toogle_state = False

            elif kernel == 3:  # RBF
                self.kernel_pca = KernelPCA(n_components=2, kernel='rbf')
                self.X_train_pca = self.kernel_pca.fit_transform(X_train)
                self.previous_kernel = kernel
                self.std_toogle_state = False

            elif kernel == 4:  # Sigmoid
                self.kernel_pca = KernelPCA(n_components=2, kernel='sigmoid')
                self.X_train_pca = self.kernel_pca.fit_transform(X_train)
                self.previous_kernel = kernel
                ex.std_toogle_state = False

            elif kernel == 5:  # Cosine
                self.kernel_pca = KernelPCA(n_components=2, kernel='cosine')
                self.X_train_pca = self.kernel_pca.fit_transform(X_train)
                self.previous_kernel = kernel
                ex.std_toogle_state = False




        model.fit(self.X_train_pca, y_train)

        if std_data == True:
            X_test_pca = self.kernel_pca.transform(X_test_std)

        else:
            X_test_pca = self.kernel_pca.transform(X_test)

        y_pred = model.predict(X_test_pca)

        ##### METRICS #####
        accuracy = sm.accuracy_score(y_test, y_pred)
        if len(np.unique(y_test)) == 2:
            precision = sm.average_precision_score(y_test, y_pred)
            recall = sm.recall_score(y_test, y_pred)
            f1_score = sm.f1_score(y_test, y_pred)
            metrics_update(accuracy, precision, recall, f1_score)

        else:
            ex.l_accuracy.setText("Model Accuracy: " + str(np.round(accuracy, decimals=3)))
            ex.l_precision.setText("")
            ex.l_recall.setText("" )
            ex.l_f1.setText("")





        ##### ACTUAL PLOTTING ###
        ax = self.figure.add_subplot(111)
        ax.clear()
        self.plot_decision_regions(self.X_train_pca, y_train, model, ax)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(loc='lower left')

        # plt.savefig('11.png', bbox_inches='tight',dpi=300)
        # start = False
        self.draw()


    def plot_decision_regions(self, X, y, classifier, ax, resolution=0.02):
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)


        ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=self.cmap)
        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())
        for idx, cl in enumerate(np.unique(y)):
            ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=self.cmap(idx), marker=self.markers[idx], label=cl)

########################################################################################################################
def load_dataset(path, train_size=0.8):
    global X_test, y_test

    # df = pd.read_csv(path, encoding="ISO-8859-1")
    df = pd.read_csv(path)

    # X, y_categoricas = df[df.columns[0:215]], df[df.columns[215]]
    X, y_categoricas = df[df.columns[col_start-1:col_end]], df[df.columns[col_label-1]]
    y = np.int32(y_categoricas)


    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0)


    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    return X_train, y_train, cmap, markers


def metrics_update(accuracy, precision, recall, f1_score):
    # print("Accuracy is: ", accuracy)
    ex.l_accuracy.setText("Model Accuracy: " + str(np.round(accuracy, decimals=3)))
    ex.l_precision.setText("Model Precision: " + str(np.round(precision, decimals=3)))
    ex.l_recall.setText("Model Recall: " + str(np.round(recall, decimals=3)))
    ex.l_f1.setText("Model F1 score: " + str(np.round(f1_score, decimals=3)))

########################################################################################################################
if __name__ == '__main__':


    ######################## GUI execution  ###########################################
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())