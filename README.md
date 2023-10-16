# 简介
主要实现通过手势进行鼠标键盘控制，进行移动、点击、拖拽/框选、上下滚动、按键触发等。

# 使用
1.先安装依赖

方式1，通过pip安装依赖（python版本为3.8.10）：
```shell
pip install -r requirement.txt
```

方式2，通过conda：
```shell
conda env create -f environment.yaml
```

2.执行主程序
```shell
python main.py
```

# 动作：

目前自定义了一些手势，有别的想法也可以修改实现

 - 移动鼠标：食指单指竖起，鼠标此时相对食指指尖在图像上的位置进行移动
 - 左键点击：食指、大拇指竖起
 - 右键点击：食指、小指竖起，或食指、无名指、小指竖起
 - 拖动/框选：食指、中指竖起，触发时会按下鼠标左键，当手势发生变化时取消按下
 - 滚轮向下：中指、无名指、小指竖起
 - 滚轮向上：食指、中指、无名指、小指竖起
 - tab：握拳+五指张开，组合手势，目前定义的为触发tab键

# 参数配置
配置参数在config.ini中，具体参数如下
```ini
[common]
# 摄像头索引，设置为-1时使用rtsp推流
index = 0
# rtsp推流地址
rtsp = rtsp://{self.user}:{self.pwd}@{self.ip}/cam/realmonitor?channel=1&subtype=0
# 摄像头的分辨率、帧率、镜像反转模式（-2表示不反转）
cap_width = 1920
cap_height = 1080
cap_fps = 60
cap_flip = -1
# 识别左手还是右手，（如果识别到多只左手或右手，会选取第一只识别到的手）
hand = Right
# 鼠标移动平滑参数
smooth = 4
# 是否显示图像，0否1是
show = 1
# 截取摄像头局部范围，左上角坐标、右下角坐标（摄像头范围大时需要截取，crop2为"0, 0"时不截取），截取范围长宽比为16:9或16:10，好映射屏幕
crop1 = 0, 0
crop2 = 0, 0
# 是否触发执行，0否，1是，为0时不会执行具体操作，用于调试
trigger = 1
```

# 打包
执行命令进行打包
```shell
pyinstaller main.py --name="hand-gesture" --add-data="config.ini;." --add-data="venv/Lib/site-packages/mediapipe/modules;mediapipe/modules" -w
```