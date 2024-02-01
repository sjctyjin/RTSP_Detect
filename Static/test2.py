import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QMenu, QAction, QDialog, QInputDialog

class TableWidgetExample(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Right-Click TableWidget Example')
        self.setGeometry(100, 100, 400, 300)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.tableWidget = CustomTableWidget()
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(['Order', 'Array'])

        data = {'1': [[760, 511]],
                '2': [[895, 499]],
                '3': [[1021, 711]],
                '4': [[829, 746]]}

        for order, array in data.items():
            row_position = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_position)

            item_order = QTableWidgetItem(order)
            item_array = QTableWidgetItem(str(array))

            self.tableWidget.setItem(row_position, 0, item_order)
            self.tableWidget.setItem(row_position, 1, item_array)

        self.layout.addWidget(self.tableWidget)

        self.central_widget.setLayout(self.layout)

class CustomTableWidget(QTableWidget):
    def __init__(self):
        super().__init__()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        change_order_action = menu.addAction('Change Order')
        action = menu.exec_(self.mapToGlobal(event.pos()))

        if action == change_order_action:
            selected_item = self.itemAt(event.pos())
            if selected_item:
                row = selected_item.row()
                current_order = self.item(row, 0).text()
                new_order, ok = QInputDialog.getText(self, 'Change Order', 'New Order:', text=current_order)

                if ok:
                    try:
                        new_order_int = int(new_order)
                        current_order_int = int(current_order)

                        # Swap the data between the selected row and the target row
                        self.swapRows(row, new_order_int - 1)

                    except ValueError:
                        pass

    def swapRows(self, row1, row2):
        # Swap data between two rows
        # for column in range(self.columnCount()):
        #     item1 = self.item(row1, column)
        #     item2 = self.item(row2, column)
        #     item1_text = item1.text()
        #     item2_text = item2.text()
        # 
        #     item1.setText(item2_text)
        #     item2.setText(item1_text)

        for column in range(self.columnCount()):
            item1 = self.item(row1, column)
            item2 = self.item(row2, column)
            item1_text = item1.text()
            item2_text = item2.text()

            item1.setText(item2_text)
            item2.setText(item1_text)

            # Swap Order values
        order1_item = self.item(row1, 0)
        order2_item = self.item(row2, 0)
        order1_text = order1_item.text()
        order2_text = order2_item.text()

        order1_item.setText(order2_text)
        order2_item.setText(order1_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TableWidgetExample()
    ex.show()
    sys.exit(app.exec_())
