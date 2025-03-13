# Downlaod the dtu dataset

# create data folder if it doesn't exist
mkdir -p data

# download from google drive
gdown 1maZGcJBFgMOsFCcKwLsw1od5Qm1ZQ2RU -O data/dtu.zip

# unzip
unzip data/dtu.zip -d data

# remove zip file
rm data/dtu.zip