# extract topics from bertopic_model_data.pkl
import pickle

PKL_PATH = ".saved_models/bertopic_model_data.pkl"

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

topic_model = data["topic_model"]
human_labels = data.get("human_labels", None)

topic_info = topic_model.get_topic_info()

# Build formatted labels from BERTopic names
raw_labels = topic_info["Name"].tolist()
formatted_labels = []
for label in raw_labels:
    parts = label.split("_")
    if parts[0] == "-1":
        clean_label = "Other / Irrelevant / Noise"
    else:
        clean_label = ", ".join(parts[1:])
    formatted_labels.append(clean_label)

topic_info["Formatted_Label"] = formatted_labels

if human_labels:
    topic_info["Human_Label"] = topic_info["Topic"].map(
        lambda tid: human_labels.get(tid, "")
    )

print(f"Total topics: {len(topic_info)}")
print()

cols = ["Topic", "Count", "Formatted_Label"]
if "Human_Label" in topic_info.columns:
    cols.append("Human_Label")

print(topic_info[cols].to_string(index=False))

# Save to CSV
out_path = "bert_topics.csv"
topic_info[cols].to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
