gitbook build ../ZinNote/
mkdir ./ZinNote/
rm -rf ./ZinNote/*
cp -r ../ZinNote/_book/ ./ZinNote/
git add .
git commit -m $1
git push
git checkout
