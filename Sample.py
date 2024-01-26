# import sys
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
# import time
# from View.sql_conn import *
# import traceback
# import json
# # from Model.MD_1 import *
# import pymssql
#
#
#
#
# class PyQt_MVC_Main(QMainWindow):
#     def __init__(self,parent=None):
#         super(QMainWindow,self).__init__(parent)
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)
#         self.setWindowTitle('PyQt 5 MVC example')
#
#         self.ui.lineEdit.setText('127.0.0.1:1433')
#         self.ui.lineEdit_2.setText('TP_LINK_ELEC')
#         self.ui.lineEdit_3.setText('sa')
#         self.ui.lineEdit_4.setText('pass')
#         # self.ui.lineEdit.setText('211.23.79.241:5603')
#         # self.ui.lineEdit_2.setText('Jongyee_iTrace')
#         # self.ui.lineEdit_3.setText('JYTest')
#         # self.ui.lineEdit_4.setText('test1234')
#         self.linkEvent()
#         self.show()
#
#
#     def linkEvent(self):
#         self.ui.SQL_Send.clicked.connect(self.onclickop)
#         return
#
#     def insert_data(self, table_widget, data):
#         row0 = data[0] if len(data) else []
#         table_widget.setRowCount(len(data))
#         table_widget.setColumnCount(len(row0))
#         for r, row in enumerate(data):
#             for c, item in enumerate(row):
#                 table_widget.setItem(r, c, QTableWidgetItem(str(item)))
#
#     def onclickop(self):
#         paramater = {}
#         # self.ui.stackedWidget.setCurrentIndex(0)
#
#
#         a = self.ui.lineEdit.text()
#         b = self.ui.lineEdit_2.text()
#         c = self.ui.lineEdit_3.text()
#         d = self.ui.lineEdit_4.text()
#
#         text = self.ui.textEdit.toPlainText()
#         paramater = {
#             "connectstring": {
#                 "Server": f"{a}",
#                 "DataBase" :f"{b}",
#                 "UserID" : f"{c}",
#                 "Pass" : f"{d}"
#
#             }
#
#         }
#         try:
#             if paramater != {}:
#                 with open('sql.json', "w") as f:
#                     json.dump(paramater, f, indent=4)  # indent : 指定縮排長度
#                     msg_box = QMessageBox(QMessageBox.Warning, '儲存', '設定檔儲存成功')
#                     msg_box.exec_()
#             else:
#                 msg_box = QMessageBox(QMessageBox.Warning, '提示', '請先完成參數設定')
#                 msg_box.exec_()
#         except:
#             msg_box = QMessageBox(QMessageBox.Warning, '儲存', '設定檔儲存失敗')
#             msg_box.exec_()
#         print(text)
#         cn =pymssql.connect(server=a, user=c, password=d, database=b, timeout=2,
#                         login_timeout=2, charset='big5')
#         cursor = cn.cursor(as_dict=True)
#         cursor.execute("SELECT TOP(3) * FROM SmartHome1")
#         data = cursor.fetchall()
#         wordd = []
#         for i in range(len(data)):
#             # wordd.append([data[i]['A01'],data[i]['A02'],data[i]['A03'],data[i]['A04'],data[i]['A05']])
#             wordd.append([data[i]['ID'],data[i]['A'],data[i]['W'],data[i]['Wh']])
#             print(data[i])
#         # self.ui.textEdit.setText(wordd)
#         self.insert_data(self.ui.tableWidget, wordd)
#         # print(wordd)
#         # print(a)
#         # print(b)
#         # print(c)
#         # print(d)
#         print("點擊開始")
#         return
#
# def main():
#     app = QtWidgets.QApplication(sys.argv)
#     main = PyQt_MVC_Main()
#     sys.exit(app.exec_())
#
# if __name__ == '__main__':
#     main()
# # import pandas as pd
# #
# # df = pd.read_csv('your_file.csv', header=None, names=['product', 'SN', 'position'])
# #
# # a = []
# # print(len(df))
# # for i in range(len(df)):
# #     a.append([df.iloc[i][0],df.iloc[i][1],df.iloc[i][2]])
# #
# # print(a[1])
from PyQt5 import QtWidgets, QtCore

class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()

class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.label = ClickableLabel("Click Me", self)
        self.label.clicked.connect(self.label_clicked)

    def label_clicked(self):
        print("Label was clicked!")

app = QtWidgets.QApplication([])
window = MyWindow()
window.show()
app.exec_()
