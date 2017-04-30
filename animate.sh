#!/bin/sh

EXP=$1
TYPE=${2:-fake-samples}
DELAY=${3:-50}

XARGS=`ls $EXP/${TYPE}*.png | \
  sed "s|$EXP/${TYPE}_||" | \
  sed "s|.png||" | \
  sort -n | \
  sed "s|^|-delay $DELAY $EXP/${TYPE}_|" | \
  sed "s|$|.png|"`

CMD="convert $XARGS $EXP/$TYPE.gif"
echo $CMD
$CMD
