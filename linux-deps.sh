#!/bin/bash
# this script only exists to install powershell in the manylinux_2_24 image
apt-get update
apt-get install -y --force-yes curl gnupg apt-transport-https
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
echo "deb [arch=amd64] https://packages.microsoft.com/repos/microsoft-debian-stretch-prod stretch main" > /etc/apt/sources.list.d/microsoft.list
apt-get update
apt-get install -y --force-yes powershell