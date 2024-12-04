#!/bin/bash

# This script deletes an EC2 instance using AWS CLI

# Setting up the instance ID that needs to be deleted
INSTANCE_ID="i-017cc5f88d0fe23b8"  

# Terminating the EC2 instance
echo "Terminating EC2 instance with ID: $INSTANCE_ID..."
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# Waiting for the instance to terminate
echo "Waiting for instance to be terminated..."
aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID

echo "EC2 instance with ID $INSTANCE_ID has been terminated."
