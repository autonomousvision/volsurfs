all:
	echo "Building volsurfs"
	pip install -v --user --editable ./ 

clean:
	pip uninstall volsurfs
	rm -rf build *.egg-info build volsurfs*.so libvolsurfs_cpp.so libvolsurfs_cu.so