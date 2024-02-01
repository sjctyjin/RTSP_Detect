import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QPushButton, QVBoxLayout, QWidget

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 400, 300)
        self.setWindowTitle('TableWidget Example')

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.tableWidget = QTableWidget()
        self.layout.addWidget(self.tableWidget)

        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(["Order", "Value"])
        self.tableWidget.setRowCount(9)

        # Populate the table with sample data
        for i in range(9):
            self.tableWidget.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(f"Value {i + 1}"))

        self.deleteButton = QPushButton("Delete Row 4")
        self.layout.addWidget(self.deleteButton)
        self.deleteButton.clicked.connect(self.deleteRow)

    def deleteRow(self):
        row_to_delete = 3  # Delete the 4th row (0-based index)

        if row_to_delete >= 0 and row_to_delete < self.tableWidget.rowCount():
            self.tableWidget.removeRow(row_to_delete)  # Step 2

            # Step 3: Update Order values for remaining rows
            for row in range(row_to_delete, self.tableWidget.rowCount()):
                order_item = QTableWidgetItem(str(row + 1))
                self.tableWidget.setItem(row, 0, order_item)

def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
