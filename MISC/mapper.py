#!/usr/bin/env python3
import sys
import gzip

def mapper():
    for line in gzip.open(sys.stdin.buffer, mode="rt"):
        print(line.strip())  # Emit each line of decompressed content

if __name__ == "__main__":
    mapper()
