#!/usr/bin/env bash

# remove build directories
sudo rm -rf build dist

# find python's site-packages path
sp_path=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
echo "$sp_path"/imageio/

# run pyinstaller with provided options
pyinstaller plenopticam_etri_09/gui/top_level.py \
    --onefile \
	--noconsole \
	--noconfirm \
	--name=plenopticam_etri_09 \
	--icon=plenopticam_etri_09/gui/icns/1055104.icns \
	--paths="$sp_path" \
  --add-data="$sp_path"/imageio/:./imageio \
  --add-data=./docs/build/html/:./docs/build/html/ \
  --exclude-module=matplotlib \
  --osx-bundle-identifier='org.pythonmac.unspecified.plenopticam_etri_09' \
  --add-binary='/usr/local/Cellar/tcl-tk/8.6.10/lib/libtk8.6.dylib':'tk' \
  --add-binary='/usr/local/Cellar/tcl-tk/8.6.10/lib/libtcl8.6.dylib':'tcl' \
  --hidden-import pkg_resources.py2_warn \
  --add-data=plenopticam_etri_09/cfg/cfg.json:cfg
#	 --add-data=/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/imageio/:./imageio \
#  --add-binary='/System/Library/Frameworks/Tk.framework/Tk':'tk' \
#  --add-binary='/System/Library/Frameworks/Tcl.framework/Tcl':'tcl' \
#  --add-binary='/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/lib/libtk8.6.dylib':'tk' \
#  --add-binary='/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/lib/libtcl8.6.dylib':'tcl' \
#  --add-data='./plenopticam_etri_09/cfg/cfg.json':'Resources/cfg' \

# extract version number from python file
version=$(sed -ne 's@__version__ = \([^]]*\)@\1@gp' plenopticam_etri_09/__init__.py)

# add config to spec file
sudo sed -i -e '$ d' ./plenopticam_etri_09.spec
echo "             bundle_identifier=None," >> ./plenopticam_etri_09.spec
echo "             info_plist={" >> ./plenopticam_etri_09.spec
echo "              'NSHighResolutionCapable': 'True'," >> ./plenopticam_etri_09.spec
echo "              'PyRuntimeLocations': $version," >> ./plenopticam_etri_09.spec
echo "              'CFBundleShortVersionString': $version," >> ./plenopticam_etri_09.spec
echo "              'CFBundleVersion': $version" >> ./plenopticam_etri_09.spec
echo "             }," >> ./plenopticam_etri_09.spec
echo "            )" >> ./plenopticam_etri_09.spec

# re-run pyinstaller with extended spec file
sudo pyinstaller plenopticam_etri_09.spec --noconfirm

#pyinstaller ./plenopticam_etri_09.spec
sudo mkdir ./dist/plenopticam_etri_09.app/Contents/Resources/cfg
sudo cp ./plenopticam_etri_09/cfg/cfg.json ./dist/plenopticam_etri_09.app/Contents/Resources/cfg/cfg.json

# grant write privileges to config file
sudo chmod -R 666 ./dist/plenopticam_etri_09.app/Contents/Resources/cfg/cfg.json

sudo cp -r ./docs ./dist/plenopticam_etri_09.app/Contents/Resources/
sudo mkdir -p ./dist/plenopticam_etri_09.app/Contents/Resources/gui/
sudo cp -r ./plenopticam_etri_09/gui/icns ./dist/plenopticam_etri_09.app/Contents/Resources/gui/

# certificate signature
sudo codesign --deep --signature-size 9400 -f -s "hahnec" ./dist/plenopticam_etri_09.app

# create dmg (requires npm and create-dmg)
#sudo xcode-select -switch "/Applications/Xcode.app/Contents/Developer/"
sudo create-dmg ./dist/plenopticam_etri_09.app ./dist

# replace space by underscore
for file in ./dist/*.dmg
do
  mv -- "$file" "${file// /_}"
done
