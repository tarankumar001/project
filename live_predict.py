"""
Live Motor Status Prediction
Reads real-time data from Arduino and predicts motor status using trained model.
"""

import serial
import time
import sys
import os
import numpy as np
import joblib

# Configuration
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600
MODEL_FILENAME = 'motor_model.pkl'

def load_model(filename):
    """
    Load the trained model from file.
    Returns: Trained model
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Error: {filename} not found. Please run train_model.py first to train the model.")
        
        model = joblib.load(filename)
        print(f"Model loaded successfully from {filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def compute_vibration_rms(ax, ay, az):
    """
    Compute vibration RMS from accelerometer data.
    Formula: sqrt(ax^2 + ay^2 + az^2)
    """
    return np.sqrt(ax**2 + ay**2 + az**2)

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

def predict_motor_status(model, temperature, vib_rms):
    """
    Predict motor status using the trained model.
    Returns: Predicted label (0 = NORMAL, 1 = FAULT, or other)
    """
    try:
        # Prepare features as 2D array (model expects this format)
        features = np.array([[temperature, vib_rms]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def get_status_label(prediction):
    """
    Convert prediction label to status string.
    Assumes: 0 = NORMAL, 1 = FAULT (or any non-zero = FAULT)
    """
    if prediction == 0:
        return "NORMAL"
    else:
        return "FAULT"

def main():
    """Main function to run live prediction."""
    print("Live Motor Status Prediction")
    print("=" * 50)
    
    # Load trained model
    try:
        model = load_model(MODEL_FILENAME)
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPlease run train_model.py first to train the model.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error loading model: {e}")
        sys.exit(1)
    
    # Connect to Arduino
    try:
        print(f"\nConnecting to Arduino on {SERIAL_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)  # Wait for Arduino to initialize
        print("Connected successfully!")
        print("Starting live prediction... (Press Ctrl+C to stop)")
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
    
    # Main loop: read data and predict
    try:
        while True:
            # Read data from Arduino
            data = read_arduino_data(ser)
            
            if data is not None:
                temperature, ax, ay, az = data
                
                # Compute vibration RMS
                vib_rms = compute_vibration_rms(ax, ay, az)
                
                # Make prediction
                prediction = predict_motor_status(model, temperature, vib_rms)
                
                if prediction is not None:
                    # Get status label
                    status = get_status_label(prediction)
                    
                    # Print prediction result
                    print(f"Temperature: {temperature:.2f}°C | "
                          f"Vibration RMS: {vib_rms:.2f} | "
                          f"Motor Status: {status}")
                else:
                    print("Prediction failed. Retrying...")
            else:
                # If read failed, wait a bit before trying again
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopping live prediction...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close serial connection
        if ser.is_open:
            ser.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    main()
