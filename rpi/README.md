# Raspberry Pi Edge Capture Agent - Setup Guide

A bird detection edge capture agent that runs on Raspberry Pi, capturing images and communicating with a remote inference server.

## Prerequisites
- Raspberry Pi (tested on RPi 5) with Raspberry Pi OS installed
- Raspberry Pi Camera Module connected
- Network connectivity (WiFi or Ethernet)
- Your local machine with SSH client

## 1. Setting up SSH Key Authentication

**On your local machine:**

Generate an SSH key pair if you don't already have one:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Copy the public key to the Raspberry Pi:
```bash
ssh-copy-id rpi5@<raspberry-pi-ip-address>
```

Or manually copy the key:
```bash
# Display your public key
cat ~/.ssh/id_ed25519.pub

# SSH into the RPI with password
ssh rpi5@<raspberry-pi-ip-address>

# On the RPI, create .ssh directory if it doesn't exist
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Add your public key to authorized_keys
echo "your-public-key-here" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Test the connection:
```bash
ssh rpi5@<raspberry-pi-ip-address>
```

## 2. Installing System Packages

SSH into your Raspberry Pi and install required system packages:

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install Python 3 and pip
sudo apt install -y python3 python3-pip python3-venv

# Install picamera2 and dependencies (RPi-specific)
sudo apt install -y python3-picamera2 python3-libcamera

# Install other system dependencies for OpenCV
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev \
    libjasper-dev libqt5gui5 libqt5test5 libqtcore4
```

## 3. Setting up Python Virtual Environment

Clone the repository and set up the virtual environment:

```bash
# Navigate to your home directory or preferred location
cd ~

# Clone the repository (adjust URL as needed)
git clone <your-repo-url> bird-detection
cd bird-detection/rpi

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies from requirements.txt
pip install -r requirements.txt
```

**Note:** On Raspberry Pi OS, `picamera2` is better installed via `apt` (as shown in step 2) rather than `pip`. If you already installed it via apt, you can skip it in requirements.txt or create a symlink to make it available in the venv:

```bash
# Create symlink to system picamera2 package (if needed)
ln -s /usr/lib/python3/dist-packages/picamera2 venv/lib/python3.11/site-packages/
ln -s /usr/lib/python3/dist-packages/libcamera venv/lib/python3.11/site-packages/
```

## 4. Verify Installation

Test that all packages are installed correctly:

```bash
# Activate virtual environment if not already active
source venv/bin/activate

# Test imports
python3 -c "import numpy; import cv2; import yaml; import flask; print('All packages imported successfully!')"

# Test picamera2 (only works on actual Raspberry Pi with camera)
python3 -c "from picamera2 import Picamera2; print('picamera2 available!')"
```

## 5. Configuration

Create or edit the configuration file:

```bash
cd ~/bird-detection/rpi
cp configs/image_capture.yaml.example configs/image_capture.yaml
nano configs/image_capture.yaml
```

Update the configuration parameters as needed (server URL, storage paths, operating hours, etc.).

## 6. Running the Application

```bash
# Activate virtual environment
source ~/bird-detection/rpi/venv/bin/activate

# Run the application
cd ~/bird-detection/rpi/src
python3 main.py --config ../configs/image_capture.yaml --log-level INFO
```

For debug output:
```bash
python3 main.py --config ../configs/image_capture.yaml --log-level DEBUG
```

## 7. Setting up as a System Service (Optional)

For production deployment, set up the application to run as a systemd service:

```bash
# Create systemd service file
sudo nano /etc/systemd/system/bird-capture.service
```

Add the following content (adjust paths as needed):
```ini
[Unit]
Description=Bird Detection Edge Capture Agent
After=network.target

[Service]
Type=simple
User=rpi5
WorkingDirectory=/home/rpi5/bird-detection/rpi/src
Environment="PATH=/home/rpi5/bird-detection/rpi/venv/bin"
ExecStart=/home/rpi5/bird-detection/rpi/venv/bin/python3 main.py --config /home/rpi5/bird-detection/rpi/configs/image_capture.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable bird-capture.service

# Start the service
sudo systemctl start bird-capture.service

# Check status
sudo systemctl status bird-capture.service

# View logs
sudo journalctl -u bird-capture.service -f
```

## Monitoring

The application exposes HTTP endpoints for monitoring:

- **Health check:** `http://<rpi-ip>:8080/health`
- **Prometheus metrics:** `http://<rpi-ip>:8080/metrics`
- **JSON statistics:** `http://<rpi-ip>:8080/stats`

Example:
```bash
curl http://localhost:8080/stats | jq
```

## Troubleshooting

### Camera not detected
```bash
# Check if camera is detected
libcamera-hello --list-cameras

# Test camera
libcamera-still -o test.jpg
```

### Permission issues
Ensure your user is in the `video` group:
```bash
sudo usermod -a -G video $USER
# Log out and back in for changes to take effect
```

### Module import errors
Make sure the virtual environment is activated before running the application:
```bash
source ~/bird-detection/rpi/venv/bin/activate
```

### Checking service logs
```bash
# View recent logs
sudo journalctl -u bird-capture.service -n 100

# Follow logs in real-time
sudo journalctl -u bird-capture.service -f

# View logs with specific priority
sudo journalctl -u bird-capture.service -p err
```

## Project Structure

```
rpi/
├── configs/
│   └── image_capture.yaml      # Configuration file
├── src/
│   ├── main.py                 # Main entry point
│   ├── capture.py              # Camera capture service
│   ├── storage.py              # Local storage management
│   ├── metrics.py              # Metrics collection and HTTP server
│   └── utils.py                # Utility functions
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Python Dependencies

The following packages are required (see `requirements.txt`):
- **numpy** (1.26.4) - Array operations
- **PyYAML** (>=6.0) - Configuration file parsing
- **requests** (>=2.28.0) - HTTP client for server communication
- **opencv-python-headless** (4.12.0.88) - Image processing
- **Flask** (>=2.3.0) - HTTP server for metrics
- **Werkzeug** (>=2.3.0) - WSGI utilities
- **picamera2** - Raspberry Pi camera interface (install via apt)
