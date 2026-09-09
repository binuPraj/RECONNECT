import json
import numpy as np

p = json.load(open("_tmp_diarization_probe_output.json", encoding="utf-8"))
for name, value in p.items():
    print(f"\n{name} turns={len(value['turns'])}")
    print("pipeline warnings:", value["clustering"]["pipeline_warnings"])
    for i, turn in enumerate(value["turns"]):
        print(i, f"{turn['start']:.6f}", f"{turn['end']:.6f}", f"{turn['end'] - turn['start']:.6f}", turn["speaker_label"], value["turn_embedding_warnings"][i]["warnings"])
    c = value["clustering"]
    print("train_count", c["training_embedding_count"])
    print("chunks", c["training_chunk_indices"])
    print("local", c["training_local_speaker_indices"])
    print("merges", " ".join(f"{x:.6f}" for x in c["ahc_merge_distances"]))
    matrix = value["turn_embedding_cosine_distance_matrix"]
    print("matrix")
    for row in matrix:
        print(" ".join("NA" if x is None else f"{x:.4f}" for x in row))
