!pip
install
roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="rXpRCBUAZtFB8pw6zBlN")
project = rf.workspace("military-drone-unabv").project("railway-defect-dataset")
版本 = 项目.版本(9)
数据集 = version.download("yolo26")
