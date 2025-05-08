# Downlaod the custom scenes

# create data folder if it doesn't exist
mkdir -p data

# download from google drive
gdown 18W9aSIL4SnCDaHJ8uaGvWF4GgAze14lm -O data/blendernerf.zip

# unzip
unzip data/blendernerf.zip -d data

# remove zip file
rm data/blendernerf.zip