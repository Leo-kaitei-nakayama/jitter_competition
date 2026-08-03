
"""Usage: python plot_loss.py experiments/asdn/train_log.csv"""
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # サーバー上でウィンドウなしで画像保存するため
import matplotlib.pyplot as plt

log_path = sys.argv[1] if len(sys.argv) > 1 else 'experiments/asdn/train_log.csv'
df = pd.read_csv(log_path)

plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('loss_curve.png')
print('Saved loss_curve.png')