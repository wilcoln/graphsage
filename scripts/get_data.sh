# Create data directory if not exists
mkdir -p data

# Clean data directory
rm -rf data/*

# Download data
curl http://snap.stanford.edu/graphsage/ppi.zip --output ppi.zip
curl http://snap.stanford.edu/graphsage/reddit.zip --output reddit.zip

# Unzip data
unzip -o ppi.zip
unzip -o reddit.zip

# Move downloaded data to data directory
mv reddit ppi data

# Clean up
rm ppi.zip reddit.zip

