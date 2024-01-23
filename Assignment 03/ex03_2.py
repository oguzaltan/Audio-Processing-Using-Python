sequences = np.load('sequences.npz')
x = sequences['x']
y1 = sequences['y1']
y2 = sequences['y2']
y3 = sequences['y3']

x_norm = x/np.max(x)
y1_norm = y1/np.max(y1)
y2_norm = y2/np.max(y2)
y3_norm = y3/np.max(y3)

y1_corr = np.max(correlate(x_norm, y1_norm))
y2_corr = np.max(correlate(x_norm, y2_norm))
y3_corr = np.max(correlate(x_norm, y3_norm))

print("Correlation of x and y1: ", y1_corr)
print("Correlation of x and y2: ", y2_corr)
print("Correlation of x and y3: ", y3_corr)

# plt.plot(x_norm,label="Original Signal")
# plt.plot(y1_norm,label="y1")
# plt.plot(y2_norm,label="y2")
# plt.plot(y3_norm,label="y3")
# plt.legend()
# plt.show()

phi_xx = correlate(voice1, voice1)
lags = correlation_lags(len(voice1), len(voice1))
plt.plot(lags, phi_xx,label="x-x")

phi_xy1 = correlate(x_norm, y1_norm)
lags_xy1 = correlation_lags(len(x_norm), len(y1_norm))
plt.plot(lags_xy1, phi_xy1,label="x-y1")

phi_xy2 = correlate(x_norm, y2_norm)
lags_xy2 = correlation_lags(len(x_norm), len(y2_norm))
plt.plot(lags_xy2, phi_xy2, label="x-y2")

phi_xy3 = correlate(x_norm, y3_norm)
lags_xy3 = correlation_lags(len(x_norm), len(y3_norm))
plt.plot(lags_xy3, phi_xy3,label="x-y3")

plt.legend()
plt.show()
