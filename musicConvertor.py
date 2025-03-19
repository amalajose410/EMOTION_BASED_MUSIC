import pandas as pd

# Load the CSV file
df = pd.read_csv('muse_v4.csv')

# Replace the API link with the normal public link
df['lastfm_url'] = df['lastfm_url'].apply(
    lambda x: x.replace('https://api.spotify.com/v1/tracks/', 'https://open.spotify.com/track/')
)

# # Save the updated DataFrame back to CSV
df.to_csv('updated_file.csv', index=False)

print("✅ Links converted and file saved as 'updated_file.csv'")
