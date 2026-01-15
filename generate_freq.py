import pandas as pd
import json
import os


def generate_item_freq(
        input_file="nyc.txt",
        output_file="item_freq.json",
        mapping_output_file="poi_mapping.json",
        min_freq=0
):
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(
        input_file,
        sep='\t',
        header=None,
        names=['User ID', 'Venue ID', 'Venue Category ID', 'Venue Category Name', 'Latitude', 'Longitude', 'Timezone',
               'UTC Time'],
        encoding='latin-1'
    )

    venue_counts = df['Venue ID'].value_counts()
    print(f"Original unique venues: {len(venue_counts)}")

    if min_freq > 0:
        venue_counts = venue_counts[venue_counts >= min_freq]
        print(f"Venues after filtering (min_freq={min_freq}): {len(venue_counts)}")


    sorted_venues = venue_counts.index.tolist()

    id_to_poi = {}
    item_freq_dict = {}

    for rank, raw_venue_id in enumerate(sorted_venues):
        poi_token = f"POI_{rank}"
        frequency = int(venue_counts[raw_venue_id])

        id_to_poi[raw_venue_id] = poi_token

        item_freq_dict[poi_token] = frequency

    print(f"Saving frequencies to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(item_freq_dict, f, indent=4)

    print(f"Saving mapping to {mapping_output_file}...")
    with open(mapping_output_file, 'w', encoding='utf-8') as f:
        json.dump(id_to_poi, f, indent=4)

    print("Done!")

    print("\nExample entries in item_freq.json:")
    print(dict(list(item_freq_dict.items())[:5]))


if __name__ == "__main__":
    # 请确保 nyc.txt 在当前目录下，或者修改路径
    if os.path.exists("nyc.txt"):
        generate_item_freq()
    else:
        print("Error: 'nyc.txt' not found. Please upload the file or check the path.")