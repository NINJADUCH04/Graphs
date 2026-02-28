data = []

for device in range(NUM_DEVICES):

    base_energy = np.random.uniform(20, 50)
    battery = np.random.uniform(60, 100)

    for t in range(TIME_STEPS):
        temperature = 25 + 5 * np.sin(0.05 * t) + np.random.normal(0, 1)
        network_load = np.random.uniform(0.3, 1.0)

        energy = (
                base_energy
                + 10 * np.sin(0.1 * t)
                + 5 * network_load
                + np.random.normal(0, 2)
        )

        battery -= 0.01 * energy

        data.append([
            device,
            t,
            energy,
            battery,
            temperature,
            network_load
        ])

df = pd.DataFrame(data, columns=[
    "device_id",
    "time",
    "energy",
    "battery",
    "temperature",
    "network_load"
])

print(df.head())