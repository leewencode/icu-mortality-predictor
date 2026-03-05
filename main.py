import subprocess
import sys

def start_app():
    print("Launching ICU Mortality Prediction Dashboard...")
    try:
        # We specify the port and address here so you don't have to type them in the terminal
        subprocess.run([
            "streamlit", "run", "script/app.py", 
            "--server.port", "8501", 
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    start_app()