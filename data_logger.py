import serial
import csv
import time
import os

# =============================
# CONFIGURATION
# =============================
PORT = "COM3"        # Change if your port is different
BAUD = 9600
CSV_FILE = "motor_data.csv"

print("Arduino Data Logger")
print(f"Connecting to Arduino on {PORT}...")

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("Connected successfully!")
except:
    print("ERROR: Could not open serial port.")
    print("Make sure:")
    print("- Arduino is connected")
    print("- Serial Monitor is CLOSED")
    print("- COM port is correct")
    exit()

print("Logging started... Press CTRL+C to stop.")

try:
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["temperature", "vibration", "label"])
        file.flush()

        while True:
            try:
                # Read temperature
                temp_line = ser.readline().decode().strip()

                # Read vibration
                vib_line = ser.readline().decode().strip()

                if temp_line and vib_line:
                    temperature = float(temp_line)
                    vibration = float(vib_line)

                    # Default label = 0 (normal)
                    writer.writerow([temperature, vibration, 0])

                    # Force save immediately
                    file.flush()

                    print(f"Saved: Temp={temperature}, Vib={vibration}")

            except ValueError:
                print("Data conversion error. Skipping line.")
            except:
                print("Serial read error.")

except KeyboardInterrupt:
    print("\nLogging stopped by user.")

finally:
    ser.close()
    print("Serial connection closed.")