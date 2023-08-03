# 简介
主要实现通过手势进行鼠标键盘控制，进行移动、点击、拖拽/框选、上下滚动、按键触发等。

# 使用
1.先安装依赖
```shell
pip install -r requirement.txt
```

2.执行主程序
```shell
python main.py
```

# 动作：
 - 移动鼠标：食指单指竖起，鼠标此时相对食指指尖在图像上的位置进行移动
 - 点击准备：食指、大拇指竖起，此时鼠标停止移动
 - 左键点击：食指、大拇指竖起并相交
 - 右键点击：食指、大拇指、中指竖起
 - 拖动/框选：食指、中指竖起，触发时会按下鼠标左键，当手势发生变化时取消按下
 - 滚轮向下：中指、无名指、小指竖起
 - 滚轮向上：食指、中指、无名指、小指竖起
 - 回车：（暂时没做，五指竖起容易误判）

# 参数配置
配置参数在config.ini中，具体参数如下
```ini
[common]
# 摄像头索引
index = 0
# 识别左手还是右手，（如果识别到多只左手或右手，会选取第一只识别到的手）
hand = Right
# 鼠标移动平滑参数
smooth = 4
# 是否显示图像，0否1是
show = 1
# 截取摄像头局部范围，左上角坐标、右下角坐标（摄像头范围大时需要截取，crop2为"0, 0"时不截取），截取范围长宽比为16:9
crop1 = 0, 0
crop2 = 0, 0
```

# 打包
可以参考package.bat的脚本进行打包