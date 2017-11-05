#!/usr/bin/env bash

# usage:
# ./ssh_port_forward host port_start port_end

cmd="ssh $1 -f -N"
for i in `seq $2 $3`; do
  cmd="$cmd -L $i:localhost:$i"
done

$cmd