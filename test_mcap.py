from rosbags.highlevel import AnyReader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types, register_types
from rosbags.typesys.msg import types_from_msg
from pathlib import Path

mcap_path = Path(r".venv/rosbags_rosbag2_2026_05_19-20_42_44_EXP-7819_rosbag2_2026_05_19-20_42_44_EXP-7819_0.mcap")

with AnyReader([mcap_path]) as reader:
    encodings = set()
    schema_names = []
    
    for connection in reader.connections:
        # For AnyReader, connections are list objects
        # We need to find schema info if available
        pass

    # In rosbags 0.11.x AnyReader handles schemas automatically for many formats
    # Let's try to print what we can find
    print(f"Number of connections: {len(reader.connections)}")
    
    count = 0
    for connection, timestamp, rawdata in reader.messages():
        if count == 0:
            print(f"First message type: {connection.msgtype}")
            try:
                msg = reader.deserialize(rawdata, connection.msgtype)
                print(f"Successfully deserialized first message")
            except Exception as e:
                print(f"Failed to deserialize first message ({connection.msgtype}): {e}")
        
        count += 1
        if count >= 200:
            break
    
    print(f"Iterated {count} messages.")
