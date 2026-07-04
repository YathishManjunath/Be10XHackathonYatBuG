# data.py
import pandas as pd

def get_messy_event_data():
    """
    Returns a DataFrame containing raw, unorganized, and conflicting speaker requests.
    Conflicts intentionally embedded:
    - Speaker 1 & 2 want the same room at the same time.
    - Speaker 3 has a time conflict with Room 102 availability.
    - Speaker 4 & 5 both require the 'High-End AV Rig' which the venue only has one of.
    """
    data = [
        {
            "Speaker Name": "Dr. Aris Thorne",
            "Topic": "Decentralized Identity Frameworks",
            "Preferred Time": "10:00 AM - 11:00 AM",
            "Preferred Venue": "Main Ballroom",
            "AV Requirements": "Dual Projectors, Lavalier Mic",
            "Speaker Availability": "10:00 AM - 1:00 PM"
        },
        {
            "Speaker Name": "Sarah Jenkins",
            "Topic": "AI Agents in Cross-Border Retail Banking",
            "Preferred Time": "10:00 AM - 11:00 AM",  # DIRECT CONFLICT WITH THORNE
            "Preferred Venue": "Main Ballroom",
            "AV Requirements": "Standard HDMI, Podiums",
            "Speaker Availability": "10:00 AM - 12:00 PM"
        },
        {
            "Speaker Name": "Prof. Nitin Kumar",
            "Topic": "Neural Network Latency Optimization",
            "Preferred Time": "11:30 AM - 12:30 PM",
            "Preferred Venue": "Room 102",  # ROOM 102 IS CLOSED FOR MAINTENANCE UNTIL 1:00 PM
            "AV Requirements": "High-End AV Rig (GPU Demo)",
            "Speaker Availability": "11:00 AM - 3:00 PM"
        },
        {
            "Speaker Name": "Elena Rostova",
            "Topic": "Scaling Web3 Architecture on Sharded Ledgers",
            "Preferred Time": "1:30 PM - 2:30 PM",
            "Preferred Venue": "Auditorium B",
            "AV Requirements": "High-End AV Rig (GPU Demo)",  # AV ASSET CONFLICT WITH NITIN KUMAR IF MOVED
            "Speaker Availability": "1:00 PM - 4:00 PM"
        },
        {
            "Speaker Name": "David Vance",
            "Topic": "Automating Regulatory Compliance via Large Action Models",
            "Preferred Time": "10:00 AM - 11:00 AM",
            "Preferred Venue": "Room 105",
            "AV Requirements": "Whiteboard, Standard Projector",
            "Speaker Availability": "10:00 AM - 11:30 AM"  # TIGHT WINDOW
        }
    ]
    return pd.DataFrame(data)

def get_venue_constraints():
    """
    Returns venue resource limits for the Agent to cross-reference.
    """
    return {
        "Available Rooms": ["Main Ballroom", "Auditorium B", "Room 105", "Conference Hall C"],
        "Room Maintenance": {"Room 102": "Unavailable before 1:00 PM"},
        "Limited Equipment": {"High-End AV Rig (GPU Demo)": "Only 1 unit available at any given time slot"}
    }