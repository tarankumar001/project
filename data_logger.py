"""
Arduino Data Logger
Reads temperature and accelerometer data from Arduino and saves to CSV file.
"""

import serial
import csv
import time
import sys

# Configuration
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600
CSV_FILENAME = 'motor_data.csv'

def setup_csv_file():
    """Create CSV file with header if it doesn't exist."""
    try:
        # Check if file exists and has header
        with open(CSV_FILENAME, 'r') as f:
            # File exists, check if header is present
            reader = csv.reader(f)
            first_line = next(reader, None)
            if first_line != ['temperature', 'ax', 'ay', 'az', 'label']:
                # Header doesn't match, we'll append it
                pass
    except FileNotFoundError:
        # File doesn't exist, create it with header
        with open(CSV_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['temperature', 'ax', 'ay', 'az', 'label'])
            print(f"Created {CSV_FILENAME} with header")

def read_arduino_data(ser):
    """
    Read two lines from Arduino:
    Line 1: temperature
    Line 2: ax,ay,az
    Returns: (temperature, ax, ay, az) or None if error
    """
    try:
        # Read first line (temperature)
        line1 = ser.readline().decode('utf-8').strip()
        if not line1:
            return None
        
        # Read second line (ax,ay,az)
        line2 = ser.readline().decode('utf-8').strip()
        if not line2:
            return None
        
        # Parse temperature
        try:
            temperature = float(line1)
        except ValueError:
            print(f"Error: Could not parse temperature from '{line1}'")
            return None
        
        # Parse accelerometer values (ax,ay,az)
        try:
            accel_values = line2.split(',')
            if len(accel_values) != 3:
                print(f"Error: Expected 3 accelerometer values, got {len(accel_values)}")
                return None
            
            ax = float(accel_values[0].strip())
            ay = float(accel_values[1].strip())
            az = float(accel_values[2].strip())
        except ValueError as e:
            print(f"Error: Could not parse accelerometer values from '{line2}': {e}")
            return None
        
        return (temperature, ax, ay, az)
    
    except serial.SerialTimeoutException:
        print("Error: Serial read timeout")
        return None
    except UnicodeDecodeError:
        print("Error: Could not decode serial data (not valid UTF-8)")
        return None
    except Exception as e:
        print(f"Error reading from Arduino: {e}")
        return None

def save_to_csv(temperature, ax, ay, az, label=0):
    """Save data row to CSV file."""
    try:
        with open(CSV_FILENAME, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([temperature, ax, ay, az, label])
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False

def main():
    """Main function to run the data logger."""
    print("Arduino Data Logger")
    print("=" * 50)
    
    # Setup CSV file with header
    setup_csv_file()
    
    # Connect to Arduino
    try:
        print(f"Connecting to Arduino on {SERIAL_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)  # Wait for Arduino to initialize
        print("Connected successfully!")
        print("Reading data... (Press Ctrl+C to stop)")
        print("-" * 50)
    except serial.SerialException as e:
        print(f"Error: Could not connect to Arduino on {SERIAL_PORT}")
        print(f"Details: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if Arduino is connected to COM3")
        print("2. Make sure no other program is using the serial port")
        print("3. Verify the baud rate matches your Arduino code")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error connecting to Arduino: {e}")
        sys.exit(1)
    
    # Main loop: read and save data
    try:
        while True:
            # Read data from Arduino
            data = read_arduino_data(ser)
            
            if data is not None:
                temperature, ax, ay, az = data
                label = 0  # Default label
                
                # Save to CSV
                if save_to_csv(temperature, ax, ay, az, label):
                    # Print saved values to console
                    print(f"Saved: Temperature={temperature:.2f}, "
                          f"ax={ax:.2f}, ay={ay:.2f}, az={az:.2f}, label={label}")
                else:
                    print("Failed to save data to CSV")
            else:
                # If read failed, wait a bit before trying again
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopping data logger...")
        print("Data saved to", CSV_FILENAME)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        # Close serial connection
        if ser.is_open:
            ser.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    main()
