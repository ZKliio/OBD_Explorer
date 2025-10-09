from asammdf import MDF

# Load the MF4 file
mdf = MDF("00000006.mf4")

# Print general info
print(mdf)

# List all available signals
print("Signals available in file:")
for i, signal in enumerate(mdf.channels_db.keys()):
    print(f"{i + 1}. {signal}")

# # Extract one signal by name (change as needed)
# signal = mdf.get("VehicleSpeed")  # Replace with actual signal name
# signal.plot()  # If matplotlib is installed, this will show a plot

# Export to CSV (optional)
mdf.export("output.csv", fmt="csv")
