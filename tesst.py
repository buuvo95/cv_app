import sys
from PyQt5.QtWidgets import QWidget, QComboBox, QApplication, QFormLayout, QLabel,QStackedLayout

class DynamicComboExample(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(300,300, 500, 300)
        self.show()

        combo_a = QComboBox(self)
        combo_a.addItem('A')
        combo_a.addItem('B')

        # set slot for when option of combobox A is changed
        combo_a.currentIndexChanged[int].connect(self.comboOptionChanged)

        self.combo_b = QComboBox()
        self.combo_b.addItem('1')
        self.combo_b.addItem('2')
        self.combo_b.addItem('3')
        self.combo_b.addItem('4')

        self.combo_c = QComboBox()
        self.combo_c.addItem('a')
        self.combo_c.addItem('b')
        self.combo_c.addItem('c')
        self.combo_c.addItem('d')
        self.combo_c.addItem('e')

        # use a stacked layout to view only one of two combo box at a time
        self.combo_container_layout = QStackedLayout()

        self.combo_container_layout.addWidget(self.combo_b)
        self.combo_container_layout.addWidget(self.combo_c)

        combo_container = QWidget()
        combo_container.setLayout(self.combo_container_layout)

        form_layout = QFormLayout()

        form_layout.addRow(QLabel('A:\t'), combo_a)

        # the stacked layout is placed in place of the (meant to be) second combobox
        form_layout.addRow(QLabel('B:\t'), combo_container)


        self.setLayout(form_layout)


    def comboOptionChanged(self, idx):
        ''' gets called when option for combobox A is changed'''

        # check which combobox_a option is selected ad show the appropriate combobox in stacked layout
        self.combo_container_layout.setCurrentIndex(0 if idx == 0 else 1)


def main():
    app= QApplication(sys.argv)
    w = DynamicComboExample()
    exit(app.exec_())

if __name__ == '__main__':
    main()