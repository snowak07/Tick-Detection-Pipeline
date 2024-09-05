## Notes
* You can't install this on Mac M1/M2 chips
* PIP dependencies are installed to /var/pip on Ubuntu 22.04

## To install on Ubuntu 22.04
### Run as Root
sudo apt-get install -y python3.10 python3.10-dev python3-distutils
sudo apt-get install -y python3-pip

### Run as owner of Apache files
curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py; python3.10 get-pip.py; rm get-pip.py
python3.10 -m pip install --upgrade pip # Necessary for upgrading PIP to v22.3+
pip cache purge; mkdir -p /var/pip; TMPDIR=/var/pip pip3.10 install --cache-dir=/var/pip --target /var/pip tensorflow==2.12 Pillow pandas numpy ultralytics

#### Test (run the following from this directory)
cp test-image.jpg ../../../test-image.jpg; python3.10 ./classify_html.py -u "http://localhost/test-image.jpg"; rm ../../../test-image.jpg

You can compare results with **test-image-results.txt**

## Installing on Docker Image
apt install -y python3.10 python3.10-dev python3-distutils
apt install -y python3-pip
mkdir -p /var/pip;
curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py; python3.10 get-pip.py; rm get-pip.py
python3.10 -m pip install --upgrade pip # Necessary for upgrading PIP to v22.3+
pip3.10 install tensorflow Pillow pandas numpy

python3.10 ./classify_html.py -u "http://localhost/lenni/TickIDNet-main/Sample Images/A. americanum/Amblyomma_americanum_f_a_iNat_unk_unfed_23674839.jpg"