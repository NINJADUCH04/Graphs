plt.figure(figsize=(8,4))

for d in range(3):  # plot first 3 devices
    subset = df[df["device_id"] == d]
    plt.plot(subset["time"], subset["energy"])

plt.title("Energy Consumption Over Time")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.show()