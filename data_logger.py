import serial
import csv
import time
import os

# =============================
# CONFIGURATION
# =============================
PORT = "COM3"        # Change if your port is different
BAUD = 9600


# =============================
# AUTO-GENERATE CSV FILENAME
# =============================
def get_csv_filename():
    """Returns the next available motor filename (motor1.csv, motor2.csv, ...)."""
    index = 1
    while True:
        name = f"motor{index}.csv"
        if not os.path.exists(name):
            return name
        index += 1


CSV_FILE = get_csv_filename()
print(f"Data will be saved to: {CSV_FILE}")


# =============================
# LABELING LOGIC
# =============================
def compute_label(temperature: float, vibration: float) -> int:
    """
    Returns 1 (fault detected) if:
      - temperature > 40  OR
      - vibration   > 60
    Returns 0 (normal) otherwise.
    """
    if temperature > 40 or vibration > 60:
        return 1
    return 0


# =============================
# SERIAL CONNECTION
# =============================
print("Arduino Data Logger")
print(f"Connecting to Arduino on {PORT}...")

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("Connected successfully!")
except Exception as e:
    print(f"ERROR: Could not open serial port. ({e})")
    print("Make sure:")
    print("  - Arduino is connected")
    print("  - Serial Monitor is CLOSED")
    print("  - COM port is correct")
    exit()


# =============================
# DATA LOGGING
# =============================
print("Logging started... Press CTRL+C to stop.")

try:
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["temperature", "vibration", "label"])
        file.flush()

        while True:
            try:
                # Read temperature and vibration from Arduino
                temp_line = ser.readline().decode().strip()
                vib_line  = ser.readline().decode().strip()

                if temp_line and vib_line:
                    temperature = float(temp_line)
                    vibration   = float(vib_line)

                    label = compute_label(temperature, vibration)

                    writer.writerow([temperature, vibration, label])
                    file.flush()

                    status = "FAULT" if label == 1 else "normal"
                    print(
                        f"Saved: Temp={temperature}°C, Vib={vibration}  "
                        f"→ label={label} ({status})"
                    )

            except ValueError:
                print("Data conversion error. Skipping line.")
            except Exception:
                print("Serial read error.")

except KeyboardInterrupt:
    print("\nLogging stopped by user.")

finally:
    ser.close()
    print("Serial connection closed.")
    print(f"Data saved to: {CSV_FILE}")