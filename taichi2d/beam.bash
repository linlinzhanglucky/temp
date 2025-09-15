#!/bin/bash

# uni test
# for i in {0..14}
# do
#     python taichi2d/beam_uni.py --batch_idx $i
# done

# region test
for i in {0..9}
do
    python beam.py --batch_idx $i &
done

# blob test
# for i in {0..10}
# do
#     python taichi2d/blob.py --batch_idx $i
# done
