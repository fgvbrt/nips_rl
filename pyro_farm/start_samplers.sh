#!/usr/bin/env bash

# usage:
#  ./start_samplers.sh host port_start num_samplers ns_host

host=$1
port_start=$2
num_samplers=$3
ns_host=$4

for i in `seq 1 $num_samplers`; do
  python sampler.py --name $HOSTNAME --host $host --port $((port_start+i-1)) --ns_host $ns_host &
done