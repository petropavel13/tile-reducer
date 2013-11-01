#!/bin/bash

tar -cvf /tmp/tile_comparer_src.tar . --exclude-vcs --exclude=tile_comparer.pro.user

scp /tmp/tile_comparer_src.tar smolin_in@172.18.36.131:/tmp/tile_comparer_src.tar

rm -r /tmp/tile_comparer_src.tar