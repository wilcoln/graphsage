# Generate docs with sphinx
cd docs
rm -rf graphsage
sphinx-apidoc -o graphsage ../graphsage
make html
cd ..