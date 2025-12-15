import sys
from PySide6.QtWidgets import QApplication
from ui.unified_main_window import UnifiedMainWindow

def main():
    app = QApplication(sys.argv)
    w = UnifiedMainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
