# Downlaod the nerf_synthetic dataset

# create data folder if it doesn't exist
mkdir -p data

# download from google drive
gdown 1OsiBs2udl32-1CqTXCitmov4NQCYdA9g -O data/blender.zip

# unzip
unzip data/blender.zip -d data

# rename nerf_synthetic to blender
mv data/nerf_synthetic data/blender

# remove zip file
rm data/blender.zip