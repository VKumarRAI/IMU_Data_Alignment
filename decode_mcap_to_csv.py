import csv
import os
import json
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from mcap.exceptions import DecoderNotFoundError
from collections import defaultdict

input_file = r".\.venv\rosbags_rosbag2_2026_05_19-20_42_44_EXP-7819_rosbag2_2026_05_19-20_42_44_EXP-7819_0.mcap"
output_dir = r".\.venv\decoded_csv"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, json.dumps(v)))
        else:
            items.append((new_key, v))
    return dict(items)

def get_msg_dict(msg):
    if msg is None: return None
    if hasattr(msg, "__slots__"):
        d = {}
        for slot in msg.__slots__:
            key = slot[1:] if slot.startswith('_') else slot
            val = getattr(msg, slot)
            d[key] = get_msg_dict(val)
        return d
    elif isinstance(msg, list):
        return [get_msg_dict(i) for i in msg]
    elif hasattr(msg, "__dict__"):
        return {k: get_msg_dict(v) for k, v in msg.__dict__.items()}
    else:
        return msg

topic_data = defaultdict(list)
topic_columns = defaultdict(set)
topic_counts = defaultdict(int)
failed_topics = set()

print(f"Reading {input_file}...")
with open(input_file, "rb") as f:
    reader = make_reader(f)
    factory = DecoderFactory()
    decoders = {}
    
    for schema, channel, message in reader.iter_messages():
        topic = channel.topic
        topic_counts[topic] += 1
        
        row = {
            "log_time": message.log_time,
            "publish_time": message.publish_time,
            "topic": topic
        }
        
        if schema:
            if schema.id not in decoders:
                try:
                    decoders[schema.id] = factory.decoder_for(schema.encoding, schema)
                except DecoderNotFoundError:
                    decoders[schema.id] = None
                    failed_topics.add(topic)
            
            decode_fn = decoders[schema.id]
            if decode_fn:
                try:
                    ros_msg = decode_fn(message.data)
                    msg_dict = get_msg_dict(ros_msg)
                    if isinstance(msg_dict, dict):
                        flat_msg = flatten_dict(msg_dict)
                        row.update(flat_msg)
                except Exception:
                    pass
        
        topic_data[topic].append(row)
        topic_columns[topic].update(row.keys())

# Write CSVs
written_files = 0
for topic, rows in topic_data.items():
    if not rows: continue
    safe_name = topic.replace("/", "__").replace(":", "__").replace(" ", "_").lstrip("__")
    csv_path = os.path.join(output_dir, f"{safe_name}.csv")
    
    base_cols = ["log_time", "publish_time", "topic"]
    other_cols = sorted(list(topic_columns[topic] - set(base_cols)))
    fieldnames = base_cols + other_cols
    
    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    written_files += 1

print(f"Summary:")
print(f"Total topics: {len(topic_counts)}")
print(f"Files written: {written_files}")
if failed_topics:
    print(f"Topics failed to decode (schema/encoding error): {len(failed_topics)}")
print("Top 10 topics by row count:")
sorted_counts = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
for t, c in sorted_counts:
    print(f"  {t}: {c}")
