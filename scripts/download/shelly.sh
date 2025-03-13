# Downlaod the shelly dataset

# create data folder if it doesn't exist
mkdir -p data

# download from google drive
gdown 1Qyf_UMd49Pm-8xjSI4j0t-Np8JWeuwOk -O data/shelly.zip

# unzip
unzip data/shelly.zip -d data

# remove zip file
rm data/shelly.zip