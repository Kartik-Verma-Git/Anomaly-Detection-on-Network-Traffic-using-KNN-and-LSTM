import pyshark
import pandas as pd
import time
import os

# Directory to save captured packets
output_dir = "captured_packets"
os.makedirs(output_dir, exist_ok=True)

def generate_filename():
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_dir, f"packets_{timestamp}.pcap")

# List to store captured packet data
captured_packets = []

def packet_callback(packet):
    global captured_packets
    try:
        packet_info = {
            "Timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "Source_IP": packet.ip.src if hasattr(packet, 'ip') else "N/A",
            "Destination_IP": packet.ip.dst if hasattr(packet, 'ip') else "N/A",
            "Protocol": packet.highest_layer,
            "Packet_Size": packet.length if hasattr(packet, 'length') else "N/A"
        }
        
        captured_packets.append(packet_info)
        print(packet_info)
    except Exception as e:
        print(f"Error processing packet: {e}")

# Start packet sniffing on wlan0 (Wi-Fi interface)
def start_sniffing(interface="wlan0", packet_count=50):
    print(f"[*] Starting packet capture on {interface}...")
    pcap_file = generate_filename()
    capture = pyshark.LiveCapture(interface=interface, output_file=pcap_file)
    
    for packet in capture.sniff_continuously(packet_count=packet_count):
        packet_callback(packet)
    
    # Save captured packets to a CSV file
    csv_file = os.path.join(output_dir, "captured_packets.csv")
    df = pd.DataFrame(captured_packets)
    df.to_csv(csv_file, index=False)
    
    print(f"[*] Packets saved to {pcap_file} and {csv_file}")

if __name__ == "__main__":
    start_sniffing()
