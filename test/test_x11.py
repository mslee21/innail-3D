import matplotlib
matplotlib.use('TkAgg')  # <--- 이 줄이 핵심입니다! (맥 X11용 백엔드)
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 1])
plt.title("Success!")
plt.show()