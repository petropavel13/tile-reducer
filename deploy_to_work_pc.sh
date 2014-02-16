#!/bin/sh

tar -cvf /tmp/tile_comparer_src.tar . --exclude-vcs --exclude=tile_comparer.pro.user

scp /tmp/tile_comparer_src.tar smolin_in@172.18.36.131:/tmp/tile_comparer_src.tar

rm /tmp/tile_comparer_src.tar

ssh smolin_in@172.18.36.131 /bin/sh <<EOF

mkdir -p ~/projects/tile_comparer

rm -f ~/projects/tile_comparer/*

tar xf /tmp/tile_comparer_src.tar -C ~/projects/tile_comparer

rm /tmp/tile_comparer_src.tar

cd ~/projects/tile_comparer

qmake tile_comparer.pro

make

EOF