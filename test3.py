from PyQt5.QtWidgets import QApplication, QMessageBox

app = QApplication([])

while True:
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Information)  # 設置圖標類型為信息圖標
    msg_box.setWindowTitle('操作提示')  # 設置對話框標題
    msg_box.setText('操作已完成，請按確定繼續。')  # 設置顯示的文本信息
    msg_box.setStandardButtons(QMessageBox.Ok)  # 只添加一個"確定"按鈕
    
    # 顯示對話框並等待用戶關閉它
    msg_box.exec_()
    
    # 對話框關閉後執行的代碼
    print("用戶已確定，繼續後續操作。")
# 這裡放置用戶點擊確定後你希望執行的代碼
# 例如：do_something()
